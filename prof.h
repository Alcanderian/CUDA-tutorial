#ifndef __4ciu7ERJN3R8n398__
#define __4ciu7ERJN3R8n398__

#include <chrono>
#include <iostream>

struct hs_timer
{
    std::chrono::time_point<std::chrono::high_resolution_clock> start;
    std::chrono::time_point<std::chrono::high_resolution_clock> end;

    void tic(const char * name)
    {
        start = std::chrono::high_resolution_clock::now();
    }

    void toc(const char * name)
    {
        end = std::chrono::high_resolution_clock::now();
        std::cout << "[" << name << " time]: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
                  << " ms\n";
    }
};

#endif