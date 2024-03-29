//RoSE Control Packet Headers
#define CS_RESET 0xff
#define CS_GRANT_TOKEN 0x80
#define CS_REQ_CYCLES 0x81
#define CS_RSP_CYCLES 0x82
#define CS_DEFINE_STEP 0x83
#define CS_RSP_STALL 0x84
#define CS_CFG_BW 0x85
#define CS_CFG_ROUTE 0x86
//RoSE Payload Packet Headers
#define CS_CAMERA 0x10
#define CS_IMU 0x20
#define CS_ACCELEROMETER 0x22
#define CS_GYROSCOPE 0x24
#define CS_ARM 0x01
#define CS_DISARM 0x02
#define CS_TAKEOFF 0x04