//See LICENSE for license details
#ifndef __AIRSIM_H
#define __AIRSIM_H

#include "serial.h"
#include <signal.h>

// COSIM-CODE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>

#include <unistd.h>
#include <netdb.h> 

#include <cstdlib>
#include <cstring>

#include <deque>
#include <mutex>


#define ROBOTICS_COSIM_BUFSIZE 1024// *1024
// COSIM-CODE

// Synchronization Commands 
#define CS_GRANT_TOKEN 0x80
#define CS_REQ_CYCLES  0x81
#define CS_RSP_CYCLES  0x82
#define CS_DEFINE_STEP 0x83
#define CS_RSP_STALL   0x84

// Data Commands
#define CS_REQ_WAYPOINT 0x01
#define CS_RSP_WAYPOINT 0x02
#define CS_SEND_IMU 0x03
#define CS_REQ_ARM  0x04
#define CS_REQ_DISARM  0x05
#define CS_REQ_TAKEOFF  0x06

typedef struct AIRSIMBRIDGEMODULE_struct {
    unsigned long out_bits;
    unsigned long out_valid;
    unsigned long out_ready;
    unsigned long in_bits;
    unsigned long in_valid;
    unsigned long in_ready;
    unsigned long in_ctrl_bits;
    unsigned long in_ctrl_valid;
    unsigned long in_ctrl_ready;
    unsigned long cycle_count;
    unsigned long cycle_budget;
    unsigned long cycle_step;
} AIRSIMBRIDGEMODULE_struct;

class cosim_packet_t
{
    public:
        cosim_packet_t();
        cosim_packet_t(uint32_t cmd, uint32_t num_bytes, char * data);
        cosim_packet_t(char * buf);
        ~cosim_packet_t();

        void print();
        void init(uint32_t cmd, uint32_t num_bytes, char * data);
        void encode(char * buf);
        void decode(char * buf);

        uint32_t cmd;
        uint32_t num_bytes;
        uint32_t * data;

};

class airsim_t: public bridge_driver_t
{
    public:
        airsim_t(simif_t& sim, AIRSIMBRIDGEMODULE_struct mmio_addrs, int airsimno, const std::vector<std::string> &args);
        ~airsim_t();
        virtual void tick();
        // Our AIRSIM bridge's initialzation and teardown procedures don't
        // require interaction with the FPGA (i.e., MMIO), and so we don't need
        // to define init and finish methods (we can do everything in the
        // ctor/dtor)
        void connect_synchronizer();
        void process_tcp_packet();
        void enqueue_firesim_data();
        void schedule_firesim_data();
        bool read_firesim_packet(cosim_packet_t * packet);
        void grant_cycles();
        void report_cycles();
        void check_stall();
        void report_stall();
        void set_step_size(uint32_t step_size);
        virtual void init() {};
        virtual void finish() {};
        // Our AIRSIM bridge never calls for the simulation to terminate
        virtual bool terminate() { return false; }
        // ... and thus, never returns a non-zero exit code
        virtual int exit_code() { return 0; }

	static char KIND;
        AIRSIMBRIDGEMODULE_struct  mmio_addrs;
        serial_data_t<uint32_t> data;

        int inputfd;
        int outputfd;
        int loggingfd;

        // COSIM-CODE
        int sync_sockfd, data_sockfd, sync_portno, data_portno, n;
        struct sockaddr_in sync_serveraddr;
        struct sockaddr_in data_serveraddr;
        struct hostent *server;
        char *hostname;
        char buf[ROBOTICS_COSIM_BUFSIZE];
    
        pthread_t tcp_thread;

        bool checking_stall;
        
        std::deque<uint32_t> fsim_rxdata;
        std::deque<uint32_t> fsim_txdata;
        std::deque<uint32_t> tcp_sync_rxdata;
        std::deque<uint32_t> tcp_data_rxdata;
        std::deque<uint32_t> tcp_txdata;
        // COSIM-CODE

        void send();
        void recv();
};

#endif // __AIRSIM_H
