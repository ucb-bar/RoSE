// See LICENSE for license details

// Note: struct_guards just as in the headers
#ifdef AIRSIMBRIDGEMODULE_struct_guard

#include "airsim.h"
#include <sys/stat.h>
#include <fcntl.h>

#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

#include <pthread.h>


/* There is no "backpressure" to the user input for sigs. only one at a time
 * non-zero value represents unconsumed special char input.
 *
 * Reset to zero once consumed.
 */

int count = 0;
int thread_count = 0;
std::mutex m;

// COSIM-CODE
// Add these functions due to overloading in the uart class
ssize_t net_write(int fd, const void *buf, size_t count)
{
    return send(fd, buf, count, 0);
}

ssize_t net_read(int fd, void *buf, size_t count)
{
    // int k = 0;
    // for(in i = 0; i < count; i++) {

    // }
    // return k;
    return recv(fd, buf, count, 0);
}
// COSIM-CODE


void * queue_func(void * arg){
    airsim_t * sim = (airsim_t *) arg;

    uint32_t cmd;
    uint32_t num_bytes;
    int n;

    std::deque<uint32_t> * curr_q;
    cosim_packet_t packet;

    FILE * txfile;
    FILE * rxfile;

    uint32_t buf[ROBOTICS_COSIM_BUFSIZE];


    // printf("[AirSim Driver Thread]: Entered thread\n");

    rxfile = fopen("/home/centos/bridge_rxdump.txt", "w");
    txfile = fopen("/home/centos/bridge_txdump.txt", "w");
    while(true) {
        thread_count++;
        if(thread_count > 50000) {
            // printf("[AIRSIM DRIVER THREAD]: Thread heartbeat\n");
            thread_count = 0;
        }
        bzero(sim->buf, ROBOTICS_COSIM_BUFSIZE);
        //usleep(1);
        n = net_read(sim->sync_sockfd, sim->buf, 4);
        if(n > 0) {
            cmd = ((uint32_t *) sim->buf)[0];
            // printf("[AIRSIM DRIVER THREAD]: Got cmd in multithreading: 0x%x, %d\n", cmd, n);
            //usleep(1);
            // if(cmd < 0x80)
                // printf("[AIRSIM DRIVER THREAD]: Got data cmd in multithreading: 0x%x\n", cmd);
            curr_q = (cmd >= 0x80) ? &(sim->tcp_sync_rxdata) : &(sim->tcp_data_rxdata);
            m.lock();
            curr_q->push_back(cmd);
            m.unlock();
            fprintf(rxfile, "%d\n", cmd);
            
            //printf("[AIRSIM DRIVER THREAD]: wrote to file\n");
            // printf("[AirSim Driver Thread]: Pushed word 0x%x\n", cmd);

            uint32_t i = 1;
            while(!net_read(sim->sync_sockfd, sim->buf, 4))
            {
                usleep(i);
                i = i * 2;
            }
            num_bytes = ((uint32_t *) sim->buf)[0];
            // printf("[AIRSIM DRIVER THREAD]: Got num_bytes in multithreading: 0x%x, %d\n", num_bytes , n);
            m.lock();
            curr_q->push_back(num_bytes);
            m.unlock();
            fprintf(rxfile, "%d\n", num_bytes);
            // printf("[AirSim Driver Thread]: Pushed word 0x%x\n", num_bytes);

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
                    // printf("[AIRSIM DRIVER THREAD]: Got datum in multithreading: 0x%x\n", ((uint32_t *) sim->buf)[i]);
                    fprintf(rxfile, "%d\n", ((uint32_t *) sim->buf)[i]);
            // printf("[AirSim Driver Thread]: Pushed word 0x%x\n", num_bytes);
                    // printf("[AirSim Driver Thread]: Pushed word 0x%x\n", ((uint32_t *) sim->buf)[i]);
                }
            }
            // printf("[AIRSIM DRIVER THREAD]: Exiting RX task\n");
        }
        if(sim->tcp_txdata.size() > 0){
            //usleep(1);
            m.lock();
            cmd = sim->tcp_txdata.front();
            sim->tcp_txdata.pop_front();
            m.unlock();
            // printf("[Airsim Driver Thread]: Got cmd from main thread: 0x%x\n", cmd);


            uint32_t i = 1;
            while(sim->tcp_txdata.size() == 0)
            {
                usleep(i);
                i = i * 2;
            }
            m.lock();
            num_bytes = sim->tcp_txdata.front();
            sim->tcp_txdata.pop_front();
            // printf("[Airsim Driver Thread]: Got num_bytes from main thread: 0x%x\n", num_bytes);

            i = 1;
            while(sim->tcp_txdata.size() < num_bytes / 4)
            {
                usleep(i);
                i = i * 2;
            }
            for(int i = 0; i < num_bytes/4; i++){
                buf[i] = sim->tcp_txdata.front();
                // printf("[Airsim Driver Thread]: Got datum from main thread: 0x%x\n", buf[i]);
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
            fprintf(txfile, "%d\n", cmd);
            fprintf(txfile, "%d\n", num_bytes);
            for(int i = 0; i < num_bytes / 4; i++) {
                fprintf(txfile, "%d\n", packet.data[i]);
            }
            // printf("[AIRSIM DRIVER THREAD]: Exiting TX task\n");
        }
    }
}
airsim_t::airsim_t(simif_t *sim, AIRSIMBRIDGEMODULE_struct *mmio_addrs, int airsimno) : bridge_driver_t(sim)
{
    // printf("[AIRSIM DRIVER] Initiated bridge driver!\n");
    this->mmio_addrs = mmio_addrs;
    this->loggingfd = 0; // unused
    this->connect_synchronizer();

    this->checking_stall = false;

    pthread_create(&(this->tcp_thread), NULL, &queue_func , this);
}

airsim_t::~airsim_t()
{
    free(this->mmio_addrs);
    close(this->loggingfd);
}

void airsim_t::connect_synchronizer()
{
    // COSIM-CODE
    // Adapted from: https://www.cs.cmu.edu/afs/cs/academic/class/15213-f99/www/class26/tcpclient.c
    this->hostname = "192.168.0.47";
    //this->hostname = "3.236.180.132";
    //this->hostname = "localhost";
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

    // this->data_sockfd = socket(AF_INET, SOCK_STREAM | SOCK_NONBLOCK, 0);
    // if (this->data_sockfd < 0)
    // {
    //     perror("ERROR opening socket");
    //     exit(0);
    // }
    // printf("Created sync socket!\n");

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

    // /* build the server's Internet address */
    // bzero((char *)&this->data_serveraddr, sizeof(this->data_serveraddr));
    // this->data_serveraddr.sin_family = AF_INET;
    // bcopy((char *)this->server->h_addr,
    //       (char *)&this->data_serveraddr.sin_addr.s_addr, this->server->h_length);
    // this->data_serveraddr.sin_port = htons(this->data_portno);
    // printf("Got data server's Internet address!\n");

    // while (connect(this->data_sockfd, (const sockaddr *)&this->data_serveraddr, sizeof(this->data_serveraddr)) < 0)
    //     ;

    printf("Connected to data server!\n");
    // COSIM-CODE
}

void airsim_t::process_tcp_packet()
{
    uint32_t cmd;
    uint32_t num_bytes;
    int n;

    uint32_t buf [ROBOTICS_COSIM_BUFSIZE];

    cosim_packet_t packet;

    if (this->tcp_sync_rxdata.size() > 0) {
        m.lock();
        cmd = this->tcp_sync_rxdata.front();
        this->tcp_sync_rxdata.pop_front();
        m.unlock();
        // printf("[Airsim Driver]: Got cmd 0x%x\n", cmd);

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
        // printf("[Airsim Driver]: Got num bytes 0x%d\n", num_bytes);

        i = 1;
        while(this->tcp_sync_rxdata.size() < num_bytes/4);
        {
            usleep(i);
            i = i * 2;
        }
        m.lock();
        for(int i = 0; i < num_bytes/4; i++){
            buf[i] = this->tcp_sync_rxdata.front();
            this->tcp_sync_rxdata.pop_front(); 
            // printf("[Airsim Driver]: Got data %d: 0x%d\n",i, buf[i]);
        }
        m.unlock();

        if(num_bytes == 0){
            packet.init(cmd, num_bytes, NULL);
        } else {
            packet.init(cmd, num_bytes, (char *) buf);
        }
        // printf("[AirSim Driver]: Got packet: ");
        // packet.print();
        switch (packet.cmd & 0xFF)
        {
        case CS_GRANT_TOKEN:
            // printf("[AirSim Driver]: Got Sync Packet\n");
            this->grant_cycles();
            // printf("[AirSim Driver]: Reporting Stalls\n");
            // this->report_stall();
            // printf("[AirSim Driver]: Reported Stalls\n");
            this->checking_stall = true;
            break;
        case CS_REQ_CYCLES:
            this->report_cycles();
            break;
        case CS_DEFINE_STEP:
            this->set_step_size(packet.data[0]);
            break;
        default:
            // TODO SEND DATA
            // if(packet.cmd < 0x80) {
            //     this->schedule_firesim_data(&packet);
            // }
            break;
        }
    }

    //bzero(this->buf, ROBOTICS_COSIM_BUFSIZE);
    //n = net_read(this->sync_sockfd, this->buf, 4);
    //if (n > 0)
    //{
    //    cmd = ((uint32_t *) buf)[0];
    //    
    //    while(!net_read(this->sync_sockfd, this->buf, 4));
    //    num_bytes = ((uint32_t *) buf)[0];
    //    if(num_bytes == 0){
    //        packet.init(cmd, num_bytes, NULL);
    //    } else {
    //        while(!net_read(this->sync_sockfd, this->buf, num_bytes));
    //        packet.init(cmd, num_bytes, this->buf);
    //    }
    //    printf("[AirSim Driver]: Got packet: ");
    //    packet.print();
    //    switch (packet.cmd & 0xFF)
    //    {
    //    case CS_GRANT_TOKEN:
    //        this->grant_cycles();
    //        break;
    //    case CS_REQ_CYCLES:
    //        this->report_cycles();
    //        break;
    //    case CS_DEFINE_STEP:
    //        this->set_step_size(packet.data[0]);
    //        break;
    //    default:
    //        // TODO SEND DATA
    //        break;
    //    }
    //}
    //printf("[AirSim Driver]: Finished processing packets\n");
}

void airsim_t::enqueue_firesim_data()
{
   do {
       this->recv();
       if(data.out.valid) {
            m.lock();
           this->fsim_rxdata.push_back(data.out.bits);
            m.unlock();
            // printf("[AIRSIM CLIENT]: ENQUEUEING PACKET -- 0x%x\n", data.out.bits);
       } else {
           break;
       }
   } while(true);
}

bool airsim_t::read_firesim_packet(cosim_packet_t * packet)
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
            for( int i = 0; i < num_bytes/4; i++ ){
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

void airsim_t::send()
{
    data.in.ready = read(this->mmio_addrs->in_ready);
    if(data.in.ready) {
        write(this->mmio_addrs->in_bits, data.in.bits);
        write(this->mmio_addrs->in_valid, 1);
    }
}

void airsim_t::recv()
{
    data.out.valid = read(this->mmio_addrs->out_valid);
    if (data.out.valid)
    {
        data.out.bits = read(this->mmio_addrs->out_bits);
        // printf("[AirSim Driver]: Got bytes %x\n", data.out.bits);
        write(this->mmio_addrs->out_ready, 1);
    }
}

void airsim_t::check_stall() 
{
    uint32_t budget;
    cosim_packet_t response;

    budget = read(this->mmio_addrs->cycle_budget);
    //printf("budget: %u\n", budget);
    if(!budget){
        response.init(CS_RSP_STALL, 0, NULL);
        // response.encode(this->buf);
        // printf("[AirSim Driver]: Sending cycles packet: ");
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

void airsim_t::report_stall()
{
    cosim_packet_t response;
    // uint32_t buf[ROBOTICS_COSIM_BUFSIZE];

    // while(read(this->mmio_addrs->cycle_budget));
    uint32_t budget;
    do {
        budget = read(this->mmio_addrs->cycle_budget);
        printf("budget: %u\n", budget);
        sleep(1);
    } while(!budget);

    response.init(CS_RSP_STALL, 0, NULL);
    // response.encode(this->buf);
    // printf("[AirSim Driver]: Sending cycles packet: ");
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

void airsim_t::grant_cycles()
{
    // printf("[AirSim Driver]: Granting Cycle\n");
    write(this->mmio_addrs->in_ctrl_bits, 1);
    write(this->mmio_addrs->in_ctrl_valid, true);
}

void airsim_t::report_cycles() 
{
    cosim_packet_t response;
    // uint32_t buf[ROBOTICS_COSIM_BUFSIZE];

    uint32_t cycles = read(this->mmio_addrs->cycle_budget);

    response.init(CS_RSP_CYCLES, 4, (char *) &cycles);
    // response.encode(this->buf);
    // printf("[AirSim Driver]: Sending cycles packet: ");
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

void airsim_t::schedule_firesim_data() {
    while(this->tcp_data_rxdata.size() > 0) {
        // printf("[AIRSIM DRIVER]: Scheduling firesim packet -- 0x%x\n", this->tcp_data_rxdata.front());
        m.lock();
        this->fsim_txdata.push_back(this->tcp_data_rxdata.front());
        this->tcp_data_rxdata.pop_front();
        m.unlock();
    }
}

void airsim_t::set_step_size(uint32_t step_size)
{
    // printf("[AirSim Driver]: Setting step size to %d!\n", step_size);
    write(this->mmio_addrs->cycle_step, step_size);
}

void airsim_t::tick()
{
    cosim_packet_t packet;
    data.out.ready = true;
    data.in.valid = false;

    count++;
    if(count > 1000) {
        // printf("[AIRSIM DRIVER]: Main heartbeat\n");
        count = 0;
    }
    
    // printf("[AirSim Driver]: Processing tick\n");
    if(this->checking_stall){
        this->check_stall();
    }
    this->process_tcp_packet();
    this->enqueue_firesim_data();
    this->schedule_firesim_data();
    if(this->read_firesim_packet(&packet)) {
        m.lock();
        this->tcp_txdata.push_back(packet.cmd);
        // printf("[AIRSIM DRIVER]: Pushing cmd %x\n", packet.cmd);
        this->tcp_txdata.push_back(packet.num_bytes);
        // printf("[AIRSIM DRIVER]: Pushing num_bytes%x\n", packet.num_bytes);
        for(int i = 0; i < packet.num_bytes/4; i++) {
            this->tcp_txdata.push_back(packet.data[i]);
            // printf("[AIRSIM DRIVER]: Pushing datum%x\n", packet.data[i]);
        }
        m.unlock();
    }
    while (this->fsim_txdata.size() > 0) {
        m.lock();
        data.in.bits = this->fsim_txdata.front();
        this->send();
        if(data.in.ready) {
            // printf("[AIRSIM DRIVER]: Transmitting firesim packet -- 0x%x\n", data.in.bits);
            this->fsim_txdata.pop_front();
            m.unlock();
        } else {
            m.unlock();
            break;
        }
    }
    //do
    //{
    //    this->recv();

    //    if (data.in.ready)
    //    {
    //        char inp;
    //        int readamt;

    //        if (data.out.fire())
    //        {
    //            printf("[AirSim Driver]: Sending data: %x\n", data.out.bits);
    //            data.in.bits = data.out.bits;
    //            data.in.valid = true;
    //        }
    //    }

    //    this->send();
    //    data.in.valid = false;
    //} while (data.in.fire() || data.out.fire());
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
    printf("cmd: 0x%x, num_bytes: %d, data: [", this->cmd, this->num_bytes);
    for (int i = 0; i < this->num_bytes / 4; i++)
        printf("%d ", this->data[i]);
    printf("]\n");
}

void cosim_packet_t::init(uint32_t cmd, uint32_t num_bytes, char *data)
{
    this->cmd = cmd;
    this->num_bytes = num_bytes;
    if (this->num_bytes > 0)
    {
        this->data = (uint32_t *)malloc(num_bytes);
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
        memcpy(this->data, &buf[8], num_bytes);
    }
}

#endif // AIRSIMBRIDGEMODULE_struct_guard
