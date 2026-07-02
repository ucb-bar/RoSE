// See LICENSE for license details

#include "rosebridge.h"
#include <sys/stat.h>
#include <fcntl.h>

#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

#include <pthread.h>
#include <queue>

char rosebridge_t::KIND;

/* There is no "backpressure" to the user input for sigs. only one at a time
 * non-zero value represents unconsumed special char input.
 *
 * Reset to zero once consumed.
 */

int count = 0;
int thread_count = 0;
std::mutex m;

ssize_t net_write(int fd, const void *buf, size_t count)
{
    return send(fd, buf, count, 0);
}

ssize_t net_read(int fd, void *buf, size_t count)
{
   return recv(fd, buf, count, 0);
}

void * queue_func(void * arg){
    printf("[ROSE BRIDGE THREAD]: Starting RX/TX thread\n");
    rosebridge_t * sim = (rosebridge_t *) arg;

    uint32_t cmd;
    uint32_t num_bytes;
    uint32_t budget;
    uint32_t big_step;

    int n;

    std::deque<uint32_t> * curr_q;
    cosim_packet_t packet;
    budget_packet_t* budget_packet; 

    uint32_t buf[ROBOTICS_COSIM_BUFSIZE];

    while(true) {
        thread_count++;
        if(thread_count > 50000) {
            // printf("[ROSE BRIDGE THREAD]: Thread heartbeat\n");
            thread_count = 0;
        }
        bzero(sim->buf, ROBOTICS_COSIM_BUFSIZE);
        //usleep(1);
        n = net_read(sim->sync_sockfd, sim->buf, 4);
        if(n > 0) {
            cmd = ((uint32_t *) sim->buf)[0];
            // printf("[ROSE BRIDGE THREAD]: Got cmd in multithreading: 0x%x, %d\n", cmd, n);
            usleep(1);
            // if(cmd < 0x80) printf("[ROSE BRIDGE THREAD]: Got data cmd in multithreading: 0x%x\n", cmd);
            curr_q = (cmd >= 0x80) ? &(sim->tcp_sync_rxdata) : &(sim->tcp_data_rxdata);
            // if this is a control sequence...
            if (cmd >= 0x80) {
                m.lock();
                curr_q->push_back(cmd);
                m.unlock();
                // fprintf(rxfile, "%d\n", cmd);
                
                // printf("[ROSE BRIDGE THREAD]: detected cmd >= 80, pushed to curr_q\n");
                // printf("[RoSE Bridge Thread]: Pushed word 0x%x\n", cmd);

                uint32_t i = 1;
                while(!net_read(sim->sync_sockfd, sim->buf, 4))
                {
                    usleep(i);
                    i = i * 2;
                }
                num_bytes = ((uint32_t *) sim->buf)[0];
                // printf("[ROSE BRIDGE THREAD]: Got num_bytes in multithreading: 0x%x, %d\n", num_bytes , n);
                m.lock();
                curr_q->push_back(num_bytes);
                m.unlock();
                // fprintf(rxfile, "%d\n", num_bytes);
                // printf("[RoSE Bridge Thread]: Pushed word 0x%x\n", num_bytes);

                if(num_bytes > 0)
                {
                    //usleep(1);
                    i = 1;
                    while(!net_read(sim->sync_sockfd, sim->buf, num_bytes))
                    {
                        usleep(i);
                        i = i * 2;
                    }
                    //usleep(1);
                    for(int i = 0; i < num_bytes / 4; i++)
                    {
                        m.lock();
                        curr_q->push_back(((uint32_t *) sim->buf)[i]);
                        m.unlock();
                        // printf("[ROSE BRIDGE THREAD]: Got datum in multithreading: 0x%x\n", ((uint32_t *) sim->buf)[i]);
                        // fprintf(rxfile, "%d\n", ((uint32_t *) sim->buf)[i]);
                        // printf("[RoSE Bridge Thread]: Pushed word 0x%x\n", num_bytes);
                        // printf("[RoSE Bridge Thread]: Pushed word 0x%x\n", ((uint32_t *) sim->buf)[i]);
                    }
                }
                // printf("[ROSE BRIDGE THREAD]: Exiting RX task\n");
            // if this is a data sequence... (cmd < 0x80)
            } else { 
                // printf("[ROSE BRIDGE THREAD]: Got data cmd in multithreading: 0x%x\n", cmd);
                uint32_t i = 1;
                while(!net_read(sim->sync_sockfd, sim->buf, 4))
                {
                    usleep(i);
                    i = i * 2;
                }
                big_step = ((uint32_t *) sim->buf)[0]; 
                i = 1;
                // printf("[ROSE BRIDGE THREAD]: Got big_step in multithreading: 0x%x\n", big_step);
                while(!net_read(sim->sync_sockfd, sim->buf, 4))
                {
                    usleep(i);
                    i = i * 2;
                }
                budget = ((uint32_t *) sim->buf)[0];
                // printf("[ROSE BRIDGE THREAD]: Got budget in multithreading: 0x%x\n", budget);
                i = 1;
                while(!net_read(sim->sync_sockfd, sim->buf, 4))
                {
                    usleep(i);
                    i = i * 2;
                }
                num_bytes = ((uint32_t *) sim->buf)[0];
                // printf("[ROSE BRIDGE THREAD]: Got num_bytes in multithreading: 0x%x\n", num_bytes);
                if(num_bytes > 0)
                {
                    //usleep(1);
                    i = 1;
                    while(!net_read(sim->sync_sockfd, sim->buf, num_bytes))
                    {
                        usleep(i);
                        i = i * 2;
                    }
                }
                // printf("[ROSE BRIDGE THREAD]: Finished receiving one packet\n");
                for (int i = 0; i < num_bytes / 4; i++) {
                    buf[i] = ((uint32_t *) sim->buf)[i];
                    // printf("[ROSE BRIDGE THREAD]: Got datum in multithreading: 0x%x\n", buf[i]);
                }
                if (num_bytes == 0) {
                    //allocate new budget_packet
                    budget_packet = new budget_packet_t(cmd, big_step, budget, num_bytes, NULL);
                } else {
                    //allocate new budget_packet
                    budget_packet = new budget_packet_t(cmd, big_step, budget, num_bytes, buf);
                }
                // printf("[ROSE BRIDGE THREAD]: Pushing budget packet %x, %x, %x\n", budget_packet.cmd, budget_packet.budget, budget_packet.num_bytes);
                m.lock();
                sim->budget_rx_queue.push_back(budget_packet);
                // printf("[ROSE BRIDGE THREAD]: current queue size: %d\n", sim->budget_rx_queue.size());
                m.unlock();
                // printf("[ROSE BRIDGE THREAD]: Pushed budget packet\n");
            }
            // printf("[ROSE BRIDGE THREAD]: Exiting RX task\n");
            // fflush(stdout);
        }
        if(sim->tcp_txdata.size() > 0){
            //usleep(1);
            m.lock();
            cmd = sim->tcp_txdata.front();
            sim->tcp_txdata.pop_front();
            m.unlock();
            if(cmd < 0x80) {
                // printf("[RoSE Driver Thread]: Got cmd from main thread: 0x%x\n", cmd);
            }
            uint32_t i = 1;
            while(sim->tcp_txdata.size() == 0)
            {
                usleep(i);
                i = i * 2;
            }
            m.lock();
            num_bytes = sim->tcp_txdata.front();
            sim->tcp_txdata.pop_front();
            // printf("[RoSE Driver Thread]: Got num_bytes from main thread: 0x%x\n", num_bytes);

            i = 1;
            while(sim->tcp_txdata.size() < num_bytes / 4)
            {
                usleep(i);
                i = i * 2;
            }
            for(int i = 0; i < num_bytes/4; i++){
                buf[i] = sim->tcp_txdata.front();
                // printf("[RoSE Driver Thread]: Got datum from main thread: 0x%x\n", buf[i]);
                sim->tcp_txdata.pop_front(); 
            }
            m.unlock();

            if(num_bytes == 0){
                packet.init(cmd, num_bytes, NULL);
            } else {
                packet.init(cmd, num_bytes, (char *) (buf));
            }
            packet.encode(sim->buf);
            net_write(sim->sync_sockfd, sim->buf, packet.num_bytes + 8);
            // printf("[ROSE BRIDGE THREAD]: Exiting TX task\n");
        }
    }
}
rosebridge_t::rosebridge_t(simif_t &sim, const ROSEBRIDGEMODULE_struct &mmio_addrs, int rosebridgeno, const std::vector<std::string> &args) : bridge_driver_t(sim, &KIND)
{
    printf("[ROSE DRIVER] Initiated bridge driver!\n");
    this->mmio_addrs = mmio_addrs;
    this->loggingfd = 0; // unused
    this->connect_synchronizer();

    this->checking_stall = false;
    
    #ifdef CAPTURE
    this->fsim_rx_capture = fopen("rosebridge_fsim_rxcapture.txt", "w");
        if (this->fsim_rx_capture == NULL) {
        fprintf(stderr, "failed to open capture file");
        exit(1);
    }
     this->fsim_tx_capture = fopen("rosebridge_fsim_txcapture.txt", "w");
        if (this->fsim_tx_capture == NULL) {
        fprintf(stderr, "failed to open capture file");
        exit(1);
    }
    #endif

    pthread_create(&(this->tcp_thread), NULL, &queue_func , this);
}
rosebridge_t::~rosebridge_t() = default;

void rosebridge_t::connect_synchronizer()
{
    // COSIM-CODE
    // Adapted from: https://www.cs.cmu.edu/afs/cs/academic/class/15213-f99/www/class26/tcpclient.c
    // this->hostname = "192.168.0.47";
    this->hostname = "localhost";
    // this->sync_portno   = 10100 + uartno;
    this->sync_portno = 10001;
    this->data_portno = 60002;
    
    printf("Starting simulation!\n");
    /* socket: create the socket */
    this->sync_sockfd = socket(AF_INET, SOCK_STREAM | SOCK_NONBLOCK, 0);
    if (this->sync_sockfd < 0)
    {
        perror("ERROR opening socket");
        exit(0);
    }
    printf("Created sync socket!\n");

    /* gethostbyname: get the server's DNS entry */
    this->server = gethostbyname(hostname);
    if (this->server == NULL)
    {
        fprintf(stderr, "ERROR, no such host as %s\n", hostname);
        exit(0);
    }
    printf("Got server's DNS entry!\n");

    /* build the server's Internet address */
    bzero((char *)&this->sync_serveraddr, sizeof(this->sync_serveraddr));
    this->sync_serveraddr.sin_family = AF_INET;
    bcopy((char *)this->server->h_addr,
          (char *)&this->sync_serveraddr.sin_addr.s_addr, this->server->h_length);
    this->sync_serveraddr.sin_port = htons(this->sync_portno);
    printf("Got sync server's Internet address!\n");

    /* connect: create a connection with the server */
    while (connect(this->sync_sockfd, (const sockaddr *)&this->sync_serveraddr, sizeof(this->sync_serveraddr)) < 0)
        ;
    printf("Connected to sync server!\n");

    // COSIM-CODE
}

void rosebridge_t::process_tcp_packet()
{
    uint32_t cmd;
    uint32_t num_bytes;
    int n;

    uint32_t buf [ROBOTICS_COSIM_BUFSIZE];

    cosim_packet_t packet;
    // printf("[RoSE Bridge]: Sync Queue Size: %d\n", this->tcp_data_rxdata.size());

    if (this->tcp_sync_rxdata.size() > 0) {
        m.lock();
        cmd = this->tcp_sync_rxdata.front();
        this->tcp_sync_rxdata.pop_front();
        m.unlock();
	//  printf("[RoSE Driver]: Got cmd from queue: 0x%x\n", cmd);

        //usleep(1);
        uint32_t i = 1;
        while(this->tcp_sync_rxdata.size() == 0)
        {
            usleep(i);
            i = i * 2;
        }
        m.lock();
        num_bytes = this->tcp_sync_rxdata.front();
        this->tcp_sync_rxdata.pop_front();
        m.unlock();
        // printf("[RoSE Driver]: Got num bytes 0x%d\n", num_bytes);

        i = 1;
        while(this->tcp_sync_rxdata.size() < num_bytes/4)
        {
            usleep(i);
            i = i * 2;
        }
        m.lock();
        for(int i = 0; i < num_bytes/4; i++){
            buf[i] = this->tcp_sync_rxdata.front();
            this->tcp_sync_rxdata.pop_front(); 
            // printf("[RoSE Driver]: Got data %d: 0x%d\n",i, buf[i]);
        }
        m.unlock();

        if(num_bytes == 0){
            packet.init(cmd, num_bytes, NULL);
        } else {
            packet.init(cmd, num_bytes, (char *) buf);
        }
        //printf("[RoSE Bridge]: Got packet: ");
        // packet.print();
        switch (packet.cmd & 0xFF)
        {
        case CS_GRANT_TOKEN:
            // printf("[RoSE Bridge]: Got Sync Packet\n");
            this->grant_cycles();
            // Iterate over budge rx queue and update all latency budgets to 0, without changing the order
            // printf("[RoSE Bridge]: Checkedoff Packets\n");
            // printf("[RoSE Bridge]: Reporting Stalls\n");
            // this->report_stall();
            // printf("[RoSE Bridge]: Reported Stalls\n");
            this->checking_stall = true;
            break;
        case CS_REQ_CYCLES:
            this->report_cycles();
            break;
        case CS_DEFINE_STEP:
            //printf("[RoSE Bridge]: Got Step Size: %d\n", packet.data[0]);

            this->set_step_size(packet.data[0]);
            break;
        case CS_CFG_BW:
            // printf("[RoSE Bridge]: Got Destination: %d\n", packet.data[0]);
            // printf("[RoSE Bridge]: Got Bandwidth: %d\n", packet.data[1]);
            this->config_bandwidth(packet.data[0], packet.data[1]);
            break;
        case CS_CFG_ROUTE:
            this->push_route(packet.data[0], packet.data[1]);
        default:
            // TODO SEND DATA
            // if(packet.cmd < 0x80) {
            //     this->schedule_firesim_data(&packet);
            // }
            break;
        }
    }

    //printf("[RoSE Bridge]: Finished processing packets\n");
}

void rosebridge_t::enqueue_firesim_data()
{
   do {
       this->recv();
       if(data.out.valid) {
            m.lock();
           this->fsim_rxdata.push_back(data.out.bits);
            m.unlock();
            // printf("[ROSE CLIENT]: ENQUEUEING PACKET -- 0x%x\n", data.out.bits);
       } else {
           break;
       }
   } while(true);
}

bool rosebridge_t::read_firesim_packet(cosim_packet_t * packet)
{
    uint32_t cmd;
    uint32_t num_bytes;
    uint32_t buf [ROBOTICS_COSIM_BUFSIZE];
    if (this->fsim_rxdata.size() >= 2) {
        // Get the cmd
        m.lock();
        cmd = this->fsim_rxdata.front();
        this->fsim_rxdata.pop_front();
        m.unlock();
        // Check the remaining entry count
        num_bytes = this->fsim_rxdata.front();
        // If there's a full packet, pop the whole thing off
        if(num_bytes == 0) {
            m.lock();
            this->fsim_rxdata.pop_front();
            packet->init(cmd, num_bytes, NULL);
            m.unlock();
            return true;
        } else if(this->fsim_rxdata.size() >= (num_bytes/4) + 1) {
            m.lock();
            this->fsim_rxdata.pop_front();
            for(int i = 0; i < num_bytes/4; i++ ){
                buf[i] = this->fsim_rxdata.front();
                this->fsim_rxdata.pop_front();
            }
            m.unlock();
            packet->init(cmd, num_bytes, (char *) buf);
            // printf("Got firesim packet: ");
            // packet->print();
            return true;
        // Otherwise, return the command
        } else {
            m.lock();
            this->fsim_rxdata.push_front(cmd);
            m.unlock();
            return false;
        }
    }
    return false;
}

void rosebridge_t::send()
{
    data.in.ready = read(this->mmio_addrs.in_ready);
    if(data.in.ready) {
        write(this->mmio_addrs.in_bits, data.in.bits);
        write(this->mmio_addrs.in_valid, 1);
    }
}

void rosebridge_t::send_budget()
{
    // printf("[ROSE DRIVER]: Try sending budget packet -- 0x%x\n", data.budget.bits);
    data.budget.ready = read(this->mmio_addrs.in_budget_ready);
    if(data.budget.ready) {
        write(this->mmio_addrs.in_budget_bits, data.budget.bits);
        write(this->mmio_addrs.in_budget_valid, 1);
    }
}

void rosebridge_t::send_bigstep()
{
    // printf("[ROSE DRIVER]: Try sending budget packet -- 0x%x\n", data.budget.bits);
    data.bigstep.ready = read(this->mmio_addrs.in_bigstep_ready);
    if(data.bigstep.ready) {
        write(this->mmio_addrs.in_bigstep_bits, data.bigstep.bits);
        write(this->mmio_addrs.in_bigstep_valid, 1);
    }
}

void rosebridge_t::recv()
{
    data.out.valid = read(this->mmio_addrs.out_valid);
    if (data.out.valid)
    {
        data.out.bits = read(this->mmio_addrs.out_bits);
        // printf("[RoSE Bridge]: Got bytes %x\n", data.out.bits);
        write(this->mmio_addrs.out_ready, 1);
    }
}

void rosebridge_t::check_stall() 
{
    uint32_t budget;
    cosim_packet_t response;

    budget = read(this->mmio_addrs.cycle_budget);
    // printf("[RoSE Bridge]:budget: %u\n", budget);
    if(budget == this->step_size){
        response.init(CS_RSP_STALL, 0, NULL);
        // response.encode(this->buf);
        // printf("[RoSE Bridge]: Sending cycles packet: ");
        // response.print();
        //net_write(this->sync_sockfd, this->buf, response.num_bytes + 8);
        m.lock();
        this->tcp_txdata.push_back(response.cmd);
        this->tcp_txdata.push_back(response.num_bytes);
        for(int i = 0; i < response.num_bytes/4; i++) {
            this->tcp_txdata.push_back(response.data[i]);
        }
        m.unlock();
        this->checking_stall = false;
    }
}

void rosebridge_t::report_stall()
{
    cosim_packet_t response;
    // uint32_t buf[ROBOTICS_COSIM_BUFSIZE];

    // while(read(this->mmio_addrs.cycle_budget));
    uint32_t budget;
    do {
        budget = read(this->mmio_addrs.cycle_budget);
        //printf("budget: %u\n", budget);
        sleep(1);
    } while(!budget);

    response.init(CS_RSP_STALL, 0, NULL);
    // response.encode(this->buf);
    // printf("[RoSE Bridge]: Sending cycles packet: ");
    // response.print();
    //net_write(this->sync_sockfd, this->buf, response.num_bytes + 8);
    m.lock();
    this->tcp_txdata.push_back(response.cmd);
    this->tcp_txdata.push_back(response.num_bytes);
    for(int i = 0; i < response.num_bytes/4; i++) {
        this->tcp_txdata.push_back(response.data[i]);
    }
    m.unlock();
}

void rosebridge_t::grant_cycles()
{
    // printf("[RoSE Bridge]: Granting Cycle\n");
    write(this->mmio_addrs.in_ctrl_bits, 1);
    write(this->mmio_addrs.in_ctrl_valid, true);
}

void rosebridge_t::report_cycles() 
{
    cosim_packet_t response;
    // uint32_t buf[ROBOTICS_COSIM_BUFSIZE];

    uint32_t cycles = read(this->mmio_addrs.cycle_budget);

    response.init(CS_RSP_CYCLES, 4, (char *) &cycles);
    // response.encode(this->buf);
    // printf("[RoSE Bridge]: Sending cycles packet: ");
    // response.print();
    //net_write(this->sync_sockfd, this->buf, response.num_bytes + 8);
    m.lock();
    this->tcp_txdata.push_back(response.cmd);
    this->tcp_txdata.push_back(response.num_bytes);
    for(int i = 0; i < response.num_bytes/4; i++) {
        this->tcp_txdata.push_back(response.data[i]);
    }
    m.unlock();
}

void rosebridge_t::schedule_firesim_data() {
    m.lock();
    // TODO: safe to not check for emptyness?
    while (!this->budget_rx_queue.empty() && ((this->fsim_txbudget.empty() && this->fsim_txdata.empty()))) {
        // printf("[ROSE DRIVER]: Entering schedule loop\n");
        this->fsim_tx_bigstep.push_back(this->budget_rx_queue.front()->big_step);
        // printf("[ROSE DRIVER]: Pushed big_step 0x%x\n", this->budget_rx_queue.front()->big_step);
        this->fsim_txbudget.push_back(this->budget_rx_queue.front()->budget);
        // printf("[ROSE DRIVER]: Pushed budget 0x%x\n", this->budget_rx_queue.front()->budget);
        this->fsim_txdata.push_back(this->budget_rx_queue.front()->cmd);
        // printf("[ROSE DRIVER]: Pushed cmd 0x%x\n", this->budget_rx_queue.front()->cmd);
        this->fsim_txdata.push_back(this->budget_rx_queue.front()->num_bytes);
        // printf("[ROSE DRIVER]: Pushed num_bytes 0x%x\n", this->budget_rx_queue.front()->num_bytes);
        if (this->budget_rx_queue.front()->num_bytes > 0) {
            for(int i = 0; i < this->budget_rx_queue.front()->num_bytes/4; i++) {
                // printf("[ROSE DRIVER]: Got buf: 0x%x\n", (this->budget_rx_queue.front()->data)[i]);
                this->fsim_txdata.push_back((this->budget_rx_queue.front()->data)[i]);
            }
        }
        // printf("[ROSE DRIVER]: Popping budget packet\n");
        // free the front to avoid leak
        delete this->budget_rx_queue.front();
        this->budget_rx_queue.pop_front();
        // printf("[ROSE DRIVER]: Popped budget packet\n");
    }
    m.unlock();

    // printf("[ROSE DRIVER]: Finished scheduling firesim data\n");
}

void rosebridge_t::set_step_size(uint32_t step_size)
{
    printf("[RoSE Bridge]: Setting step size to %d!\n", step_size);
    write(this->mmio_addrs.cycle_step, step_size);
    this->step_size = step_size;
}

void rosebridge_t::config_bandwidth(uint32_t dest, uint32_t bandwidth)
{
    printf("[RoSE Bridge]: Setting bandwidth to %d!\n", bandwidth);
    write(this->mmio_addrs.bww_config_destination, dest);
    write(this->mmio_addrs.bww_config_bits, bandwidth);
    write(this->mmio_addrs.bww_config_valid, 1);
}

// void rosebridge_t::config_route(uint32_t header, uint32_t channel)
// {
//     printf("[RoSE Bridge]: Setting header to 0x%x and channel to %d!\n", header, channel);
//     write(this->mmio_addrs.config_routing_header, header);
//     write(this->mmio_addrs.config_routing_channel, channel);
//     write(this->mmio_addrs.config_routing_valid, 1);
// }
void rosebridge_t::push_route(uint32_t header, uint32_t channel)
{
    printf("[RoSE Bridge]: pushing header to 0x%x and channel to %d!\n", header, channel);
    m.lock();
    this->fsim_cfg_header.push_back(header);
    this->fsim_cfg_channel.push_back(channel);
    m.unlock();
}

void rosebridge_t::send_route()
{
    data.cfg_routing.ready = read(this->mmio_addrs.config_routing_ready);
    if(data.cfg_routing.ready) {
        write(this->mmio_addrs.config_routing_header, data.cfg_routing.header);
        write(this->mmio_addrs.config_routing_channel, data.cfg_routing.channel);
        write(this->mmio_addrs.config_routing_valid, 1);
    }
}

void rosebridge_t::tick()
{
    cosim_packet_t packet;
    data.out.ready = true;
    data.in.valid = false;

    count++;
    if(count>1000) {
        count = 0;
    }
    
    // printf("[RoSE Bridge]: Processing tick\n");
    if(this->checking_stall){
        this->check_stall();
    }
    this->process_tcp_packet();
    this->enqueue_firesim_data();
    this->schedule_firesim_data();
    // printf("[ROSE DRIVER]: Finished scheduling loop\n");
    if(this->read_firesim_packet(&packet)) {
        m.lock();
        this->tcp_txdata.push_back(packet.cmd);
        printf("[ROSE DRIVER]: Pushing cmd %x\n", packet.cmd);
        this->tcp_txdata.push_back(packet.num_bytes);
        // printf("[ROSE DRIVER]: Pushing num_bytes%x\n", packet.num_bytes);
        for(int i = 0; i < packet.num_bytes/4; i++) {
            this->tcp_txdata.push_back(packet.data[i]);
            #ifdef CAPTURE
            for (int j = 0; j < 32; j+=8){
                fprintf(this->fsim_rx_capture, "%02x ", (packet.data[i] >> j) & 0xFF);
            }
            #endif
            // printf("[ROSE DRIVER]: Pushing datum%x\n", packet.data[i]);
        }
        #ifdef CAPTURE
            fputc('\n', this->fsim_rx_capture);
            fflush(this->fsim_rx_capture);
        #endif
        m.unlock();
    }
    while(this->fsim_tx_bigstep.size() > 0){
        // printf("[ROSE DRIVER]: Entered budget loop\n");
        m.lock();
        data.bigstep.bits = this->fsim_tx_bigstep.front();
        this->send_bigstep();
        if(data.bigstep.ready) {
            this->fsim_tx_bigstep.pop_front();
            m.unlock();
        } else {
            m.unlock();
            break;
        }
        // printf("[ROSE DRIVER]: I tried");
    }
    while(this->fsim_cfg_header.size() > 0 && this->fsim_cfg_channel.size() > 0){
        // printf("[ROSE DRIVER]: Entered budget loop\n");
        m.lock();
        data.cfg_routing.header = this->fsim_cfg_header.front();
        data.cfg_routing.channel= this->fsim_cfg_channel.front();
        this->send_route();
        if(data.cfg_routing.ready) {
            // printf("[ROSE DRIVER]: Transmitting firesim budget -- 0x%x\n", data.budget.bits);
            this->fsim_cfg_header.pop_front();
            this->fsim_cfg_channel.pop_front();
            m.unlock();
        } else {
            m.unlock();
            break;
        }
        // printf("[ROSE DRIVER]: I tried");
    }
    while(this->fsim_txbudget.size() > 0){
        // printf("[ROSE DRIVER]: Entered budget loop\n");
        m.lock();
        data.budget.bits = this->fsim_txbudget.front();
        this->send_budget();
        if(data.budget.ready) {
            this->fsim_txbudget.pop_front();
            m.unlock();
        } else {
            m.unlock();
            break;
        }
        // printf("[ROSE DRIVER]: I tried");
    }
    while (this->fsim_txdata.size() > 0) {
        m.lock();
        data.in.bits = this->fsim_txdata.front();
        this->send();
        if(data.in.ready) {
	    #ifdef CAPTURE
            fprintf(this->fsim_tx_capture, "%08x ", data.in.bits);
            fputc('\n', this->fsim_tx_capture);
            fflush(this->fsim_tx_capture);
            #endif
            this->fsim_txdata.pop_front();
            m.unlock();
        } else {
            m.unlock();
            break;
        }
    }
}


budget_packet_t::budget_packet_t(){
    this->cmd = 0x00;
    this->budget = 0;
    this->big_step = 0;
    this->num_bytes = 0;
    this->data = NULL;
}

budget_packet_t::budget_packet_t(uint32_t cmd, uint32_t big_step, uint32_t budget, uint32_t num_bytes, uint32_t * data)
{
    this->cmd = cmd;
    this->budget = budget;
    this->big_step = big_step;
    this->num_bytes = num_bytes;
    if (this->num_bytes > 0)
    {
        this->data = (uint32_t *)malloc(num_bytes);
        if (NULL == this->data){
            printf("malloc failed");
            exit(-1);
        }
        memcpy(this->data, data, num_bytes);
    }
}

budget_packet_t::~budget_packet_t()
{
    if(this->data != NULL){
        free(this->data);
    }
}

cosim_packet_t::cosim_packet_t()
{
    this->cmd = 0x00;
    this->num_bytes = 0;
    this->data = NULL;
}

cosim_packet_t::cosim_packet_t(uint32_t cmd, uint32_t num_bytes, char *data)
{
    this->init(cmd, num_bytes, data);
}

cosim_packet_t::cosim_packet_t(char *buf)
{
    this->decode(buf);
}

cosim_packet_t::~cosim_packet_t()
{
    if(this->data != NULL){
        free(this->data);
    }
}

void cosim_packet_t::print()
{
    //printf("cmd: 0x%x, num_bytes: %d, data: [", this->cmd, this->num_bytes);
    for (int i = 0; i < this->num_bytes / 4; i++)
        printf("%d ", this->data[i]);
    //printf("]\n");
}

void cosim_packet_t::init(uint32_t cmd, uint32_t num_bytes, char *data)
{
    this->cmd = cmd;
    this->num_bytes = num_bytes;
    if (this->num_bytes > 0)
    {
        this->data = (uint32_t *)malloc(num_bytes);
        if (NULL == this->data){
            printf("malloc failed");
            exit(-1);
        }
        memcpy(this->data, data, num_bytes);
    }
}

void cosim_packet_t::encode(char *buf)
{
    memcpy(&buf[0], &(this->cmd), sizeof(uint32_t));
    memcpy(&buf[4], &(this->num_bytes), sizeof(uint32_t));
    if (this->num_bytes > 0)
    {
        memcpy(&buf[8], this->data, this->num_bytes);
    }
}

void cosim_packet_t::decode(char *buf)
{
    memcpy(&(this->cmd), &buf[0], sizeof(uint32_t));
    memcpy(&(this->num_bytes), &buf[4], sizeof(uint32_t));
    if (this->num_bytes > 0)
    {
        this->data = (uint32_t *)malloc(num_bytes);
        if (NULL == this->data){
            printf("malloc failed 2");
            exit(-1);
        }
        memcpy(this->data, &buf[8], num_bytes);
    }
}
