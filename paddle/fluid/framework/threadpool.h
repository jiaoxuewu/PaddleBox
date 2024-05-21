/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <condition_variable>  // NOLINT
#include <functional>
#include <future>  // NOLINT
#include <memory>
#include <mutex>  // NOLINT
#include <queue>
#include <thread>  // NOLINT
#include <utility>
#include <vector>

#include "glog/logging.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/macros.h"  // for DISABLE_COPY_AND_ASSIGN

namespace paddle {
namespace framework {

struct ExceptionHandler {
  mutable std::future<std::unique_ptr<platform::EnforceNotMet>> future_;
  explicit ExceptionHandler(
      std::future<std::unique_ptr<platform::EnforceNotMet>>&& f)
      : future_(std::move(f)) {}
  void operator()() const {
    auto ex = this->future_.get();
    if (ex != nullptr) {
      PADDLE_THROW(platform::errors::Fatal(
          "The exception is thrown inside the thread pool. You "
          "should use RunAndGetException to handle the exception."
          "The exception is:\n %s.",
          ex->what()));
    }
  }
};

// ThreadPool maintains a queue of tasks, and runs them using a fixed
// number of threads.
class ThreadPool {
 public:
  explicit ThreadPool(int num_threads);

  using Task = std::packaged_task<std::unique_ptr<platform::EnforceNotMet>()>;

  // Returns the singleton of ThreadPool.
  static ThreadPool* GetInstance();

  ~ThreadPool();

  // Run pushes a function to the task queue and returns a std::future
  // object. To wait for the completion of the task, call
  // std::future::wait().
  template <typename Callback>
  std::future<void> Run(Callback fn) {
    auto f = this->RunAndGetException(fn);
    return std::async(std::launch::deferred, ExceptionHandler(std::move(f)));
  }

  template <typename Callback>
  std::future<std::unique_ptr<platform::EnforceNotMet>> RunAndGetException(
      Callback fn) {
    Task task([fn]() -> std::unique_ptr<platform::EnforceNotMet> {
      try {
        fn();
      } catch (platform::EnforceNotMet& ex) {
#if defined(PADDLE_WITH_XPU) || defined(PADDLE_WITH_XPU_KP)
        VLOG(0) << "ThreadPool get EnforceNotMet exception abort:" << ex.what();
        std::abort();
#endif
        return std::unique_ptr<platform::EnforceNotMet>(
            new platform::EnforceNotMet(ex));
      } catch (const std::exception& e) {
#if defined(PADDLE_WITH_XPU) || defined(PADDLE_WITH_XPU_KP)
        VLOG(0) << "ThreadPool get unknow exception abort:" << e.what();
        std::abort();
#endif
        PADDLE_THROW(platform::errors::Fatal(
            "Unexpected exception is catched in thread pool. All "
            "throwable exception in Paddle should be an EnforceNotMet."
            "The exception is:\n %s.",
            e.what()));
      }
      return nullptr;
    });
    std::future<std::unique_ptr<platform::EnforceNotMet>> f = task.get_future();
    {
      std::unique_lock<std::mutex> lock(mutex_);
      if (!running_) {
        PADDLE_THROW(platform::errors::Unavailable(
            "Task is enqueued into stopped ThreadPool."));
      }
      tasks_.push(std::move(task));
    }
    scheduled_.notify_one();
    return f;
  }
  // binding cpu cores
  void SetCPUAffinity(const std::vector<int>& cores, bool one_by_one = false) {
    if (cores.empty()) {
      return;
    }
    size_t core_num = cores.size();
    cpu_set_t mask;
    CPU_ZERO(&mask);
    if (one_by_one) {
      for (size_t i = 0; i < threads_.size(); ++i) {
        CPU_SET(cores[i % core_num], &mask);
        pthread_setaffinity_np(threads_[i]->native_handle(), sizeof(mask),
                               &mask);
      }
    } else {
      for (size_t i = 0; i < core_num; ++i) {
        CPU_SET(cores[i], &mask);
      }
      for (size_t i = 0; i < threads_.size(); ++i) {
        pthread_setaffinity_np(threads_[i]->native_handle(), sizeof(mask),
                               &mask);
      }
    }
    // VLOG(0) << "binding read ins thread_id = " << tid << ", cpunum = " <<
  }

 private:
  DISABLE_COPY_AND_ASSIGN(ThreadPool);

  // The constructor starts threads to run TaskLoop, which retrieves
  // and runs tasks from the queue.
  void TaskLoop();

  // Init is called by GetInstance.
  static void Init();

 private:
  static std::unique_ptr<ThreadPool> threadpool_;
  static std::once_flag init_flag_;

  std::vector<std::unique_ptr<std::thread>> threads_;

  std::queue<Task> tasks_;
  std::mutex mutex_;
  bool running_;
  std::condition_variable scheduled_;
};

class ThreadPoolIO : ThreadPool {
 public:
  static ThreadPool* GetInstanceIO();
  static void InitIO();

 private:
  // NOTE: threadpool in base will be inhereted here.
  static std::unique_ptr<ThreadPool> io_threadpool_;
  static std::once_flag io_init_flag_;
};

// Run a function asynchronously.
// NOTE: The function must return void. If the function need to return a value,
// you can use lambda to capture a value pointer.
template <typename Callback>
std::future<void> Async(Callback callback) {
  return ThreadPool::GetInstance()->Run(callback);
}

template <typename Callback>
std::future<void> AsyncIO(Callback callback) {
  return ThreadPoolIO::GetInstanceIO()->Run(callback);
}

inline paddle::framework::ThreadPool* get_thread_pool(int thread_num) {
  thread_local std::shared_ptr<paddle::framework::ThreadPool> thread_pool =
      nullptr;
  if (thread_pool == nullptr) {
    thread_pool.reset(new paddle::framework::ThreadPool(thread_num));
  }
  return thread_pool.get();
}
inline void split_region(size_t all_num, size_t region_num, size_t region_index,
                         size_t* start_index, size_t* end_index) {
  size_t divisor = all_num / region_num;
  size_t remainder = all_num % region_num;
  if (region_index < remainder) {
    *start_index = (divisor + 1) * region_index;
    *end_index = *start_index + (divisor + 1);
  } else {
    *start_index = divisor * region_index + remainder;
    *end_index = *start_index + divisor;
  }
  if (*end_index > all_num) {
    *end_index = all_num;
  }
}
template <class THREAD_FUNC>
inline void parallel_run_range(size_t n, THREAD_FUNC&& func, int thread_num = 20) {
  paddle::framework::ThreadPool* thrgrp = get_thread_pool(thread_num);
  std::vector<std::future<void>> wait_futures;
  for (int tid = 0; tid < thread_num; ++tid) {
    wait_futures.emplace_back(thrgrp->Run([n, tid, thread_num, &func](void) {
      size_t start = 0;
      size_t end = 0;
      split_region(n, thread_num, tid, &start, &end);
      func(tid, start, end);
    }));
  }
  for (int i = 0; i < thread_num; ++i) {
    wait_futures[i].get();
  }
}
template <class THREAD_FUNC>
inline void parallel_run_dynamic(size_t n, THREAD_FUNC&& func, int thread_num = 20) {
  paddle::framework::ThreadPool* thrgrp = get_thread_pool(thread_num);
  std::vector<std::future<void>> wait_futures;
  std::atomic<size_t> counter(0);
  for (int tid = 0; tid < thread_num; ++tid) {
    wait_futures.emplace_back(thrgrp->Run([n, &counter, &func](void) {
      size_t i = counter++;
      while (i < n) {
        func(i);
        i = counter++;
      }
    }));
  }
  for (int i = 0; i < thread_num; ++i) {
    wait_futures[i].get();
  }
}

}  // namespace framework
}  // namespace paddle
