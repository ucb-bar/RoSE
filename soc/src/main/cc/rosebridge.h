//See LICENSE for license details

#ifndef __SERIAL_COSMO_DATA_H
#define __SERIAL_COSMO_DATA_H

template <class T>
struct serial_cosmo_data_t {
  struct {
    T bits;
    bool valid;
    bool ready;
    bool fire() { return valid && ready; }
  } in;
  struct {
    T bits;
    bool valid;
    bool ready;
    bool fire() { return valid && ready; }
  } budget;
  struct {
    T bits;
    bool ready;
    bool valid;
    bool fire() { return valid && ready; }
  } out;
  struct {
    T bits;
    bool ready;
    bool valid;
    bool fire() { return valid && ready; }
  } bigstep;
  struct {
    T header;
    T channel;
    bool ready;
    bool valid;
    bool fire() { return valid && ready; }
  } cfg_routing;


};
#endif // __SERIAL_COSMO_DATA_H

#ifndef __ROSEBRIDGE_H
#define __ROSEBRIDGE_H

#include "bridges/serial_data.h"
#include "core/bridge_driver.h"
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
#include <vector>
#include <cstdint>

#define ROBOTICS_COSIM_BUFSIZE (1024*1024)
// COSIM-CODE

// Synchronization Commands 
#define CS_GRANT_TOKEN 0x80
#define CS_REQ_CYCLES  0x81
#define CS_RSP_CYCLES  0x82
#define CS_DEFINE_STEP 0x83
#define CS_RSP_STALL   0x84
#define CS_CFG_BW      0x85
#define CS_CFG_ROUTE   0x86

// for capturing packets
#define CAPTURE

struct ROSEBRIDGEMODULE_struct {
    uint64_t out_bits;
    uint64_t out_valid;
    uint64_t out_ready;
    uint64_t in_bits;
    uint64_t in_valid;
    uint64_t in_ready;
    uint64_t in_bigstep_bits;
    uint64_t in_bigstep_valid;
    uint64_t in_bigstep_ready;
    uint64_t in_budget_bits;
    uint64_t in_budget_valid;
    uint64_t in_budget_ready;
    uint64_t in_ctrl_bits;
    uint64_t in_ctrl_valid;
    uint64_t in_ctrl_ready;
    uint64_t cycle_count;
    uint64_t cycle_budget;
    uint64_t cycle_step;
    uint64_t bww_config_bits;
    uint64_t bww_config_valid;
    uint64_t bww_config_destination;
    uint64_t config_routing_header;
    uint64_t config_routing_channel;
    uint64_t config_routing_valid;
    uint64_t config_routing_ready;
    uint64_t arb_counter_state_sheader;
    uint64_t arb_counter_budget_fired;
    uint64_t arb_counter_tx_fired;
    uint64_t arb_counter_rx_0_fired;
    uint64_t arb_counter_rx_1_fired;
    // Arbiter-stall diagnostics (must match the genROReg order in RoSEBridgeModule.scala
    // and the auto-generated FireSim-generated.const.h ROSEBRIDGEMODULE_struct offsets).
    uint64_t arb_counter_idle_rxstall;
    uint64_t arb_counter_idle_advstall;
    uint64_t arb_counter_load_rxstall;
    // Held-valid enqueue status (word-drop fix). High while a host-written word awaits
    // acceptance by the skid enq; send() polls it to guarantee drop-free delivery.
    // Appended last to match the genROReg order in RoSEBridgeModule.scala.
    uint64_t in_valid_pending;
};

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

class budget_packet_t
{
    public:
        budget_packet_t();
        budget_packet_t(uint32_t cmd, uint32_t big_step, uint32_t budget, uint32_t num_bytes, uint32_t * data);
        ~budget_packet_t();

        void print();

        uint32_t big_step;
        uint32_t budget;
        uint32_t cmd;
        uint32_t num_bytes;
        uint32_t * data;
};

struct CompareBudget
{
    bool operator()(budget_packet_t const& p1, budget_packet_t const& p2)
    {
        // return "true" if "p1" is ordered before "p2", for example:
        return p1.budget > p2.budget;
    }
};

// A streaming bridge driver: the host->FPGA bulk data (rxfifo feed) can travel
// over a 512b host-managed StreamFromHostCPU instead of per-word MMIO. The
// StreamEngine& and the from-CPU stream idx/depth are supplied by the generated
// constructor (genConstructor(..., hasStreams=true)). The actual data path is
// selected at compile time by ROSE_DMA_RX (must match the RTL's ROSE_DMA_RX
// elaboration env var); budget/bigstep/grant/config/recv stay on MMIO in both.
class rosebridge_t final: public streaming_bridge_driver_t{
    public:
      rosebridge_t(simif_t &sim, StreamEngine &stream, const ROSEBRIDGEMODULE_struct &mmio_addrs, int rosebridgeno, const std::vector<std::string> &args, int stream_from_cpu_idx, int stream_from_cpu_depth);
      ~rosebridge_t();
      virtual void tick();
      // Our ROSE bridge's initialzation and teardown procedures don't
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
      void config_bandwidth(uint32_t dest, uint32_t bandwidth);
      void push_route(uint32_t header, uint32_t channel);
      virtual void init() {};
      virtual void finish() {};
      // Our ROSE bridge never calls for the simulation to terminate
      virtual bool terminate() { return false; }
      // ... and thus, never returns a non-zero exit code
      virtual int exit_code() { return 0; }

	    static char KIND;
      ROSEBRIDGEMODULE_struct mmio_addrs;
      serial_cosmo_data_t<uint32_t> data;

      FILE *fsim_tx_capture;
      FILE *fsim_rx_capture;
      int inputfd;
      int outputfd;
      int loggingfd;

      uint32_t step_size;
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
      std::deque<uint32_t> fsim_txbudget;
      std::deque<uint32_t> fsim_cfg_header;
      std::deque<uint32_t> fsim_cfg_channel;;
      std::deque<uint32_t> fsim_tx_bigstep;
      std::deque<uint32_t> tcp_sync_rxdata;
      std::deque<uint32_t> tcp_data_rxdata;
      std::deque<uint32_t> tcp_txdata;
      std::deque<budget_packet_t*> budget_rx_queue; 
      // std::priority_queue<budget_packet_t, std::vector<budget_packet_t>, CompareBudget> budget_rx_queue; 
      // COSIM-CODE

      void send();
      void send_budget();
      void send_bigstep();
      void send_route();

      void recv();

      // Host->FPGA DMA stream (Phase 1). stream_from_cpu_idx/depth come from the
      // generated constructor. dma_pending holds the framed 512b beats for the
      // in-flight data packet; dma_pending_off tracks how many bytes have been
      // push()-ed so far (incremental, non-blocking across ticks).
      int stream_from_cpu_idx;
      int stream_from_cpu_depth;
      std::vector<uint8_t> dma_pending;
      size_t dma_pending_off;

      // Arbiter-delivery trace (ROSE_ARB_TRACE): pinpoints WHERE a reqrsp response
      // dies during a gate-nav flight hang. Reads the in-bitstream genROReg arbiter
      // counters (no bitstream rebuild). last_sched_* records the most recently
      // scheduled reqrsp response so the heartbeat can correlate a counter flatline
      // with the exact cmd/payload that stuck the arbiter (e.g. 0x42 lowdim ch2).
      uint32_t last_sched_cmd = 0;
      uint32_t last_sched_nb = 0;
      int arb_trace = -1;         // -1 = unread; 0 = off; N = heartbeat every N ticks
      uint64_t arb_hb = 0;

      // Host->FPGA rxfifo datapath, selected at runtime from ROSE_DMA_RX (was a
      // compile-time #ifdef). true = 512b DMA stream; false = per-word MMIO. MUST
      // match the bitstream's RTL (a WithRoseDmaRx bitstream needs dma_rx=true).
      bool dma_rx = false;
};

#endif // __ROSEBRIDGE_H
