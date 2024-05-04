#pragma once

#include <vector>
#include <future>
#include <thread>
#include <queue>
#include <iostream>
#include <mutex>
#include <condition_variable>

#include "crc.h"

class ThreadPool
{
private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;

    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop;


public:
    explicit ThreadPool(size_t);
    template <class F, class... Args>
    auto enqueue(F &&f, Args &&...args)
        -> std::future<typename std::result_of<F(Args...)>::type>;
    ~ThreadPool();

};


class test{
    public:
        void get_weight_distribution_direct(CRC *c);
};