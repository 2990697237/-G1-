/*
 * 程序入口：搭建控制器所需的共享对象，并以固定周期反复执行状态机。
 *
 * 建议把本文件理解为控制程序的“总开关”，它本身不计算机器人动作：
 *   main -> ControlFrame::run -> FSM::run -> 当前 FSMState::run
 *        -> CtrlComponents::sendRecv -> IOSDK::sendRecv
 */

// C++ 标准库：日志、字符串、信号处理等通用能力。
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <unistd.h>
#include <csignal>
#include <sched.h>
#include <iomanip>
#include <vector>
#include <cstring>

// OpenSSL 与网络相关头文件目前没有在 main() 中直接使用，属于项目遗留依赖；
// 保留原样，阅读入口流程时可以先跳过。
#include <openssl/sha.h>
#include <openssl/rsa.h>
#include <openssl/pem.h>
#include <openssl/bio.h>
#include <openssl/err.h>
#include <openssl/evp.h>
#include <openssl/buffer.h>
#include <sys/socket.h>
#include <sys/ioctl.h>
#include <net/if.h>
#include <netinet/in.h>
#include <arpa/inet.h>

// 项目自身的三层对象：
// ControlFrame 是外层控制门面，CtrlComponents 保存共享数据，IOSDK 对接 SDK 通信。
#include "control/ControlFrame.h"
#include "control/CtrlComponents.h"
#include "interface/IOSDK.h"

// 主循环运行标志。SIGINT（Ctrl+C）处理函数和 CtrlComponents 都持有/访问它。
bool running = true;  


// Ctrl+C 的信号回调：这里只修改标志，让主循环自然退出并执行析构清理。
// sig 是信号编号；当前实现无需读取它。
void ShutDown(int sig) 
{
    std::cout << "stop the controller" << std::endl;
    running = false;
}

// 尝试把当前进程设为 FIFO 实时调度，并使用该策略允许的最高优先级。
// 这能减少控制周期被普通进程抢占的概率；权限不足时只报错，程序仍继续运行。
void setProcessScheduler()  // 实时调度设置
{
    // getpid() 得到当前控制进程，而不是某个单独线程。
    pid_t pid = getpid();
    sched_param param;
    param.sched_priority = sched_get_priority_max(SCHED_FIFO);
    if (sched_setscheduler(pid, SCHED_FIFO, &param) == -1)
    {
        std::cout << "[ERROR] Function setProcessScheduler failed." << std::endl;
    }
}

// argc/argv 当前未参与配置解析；模型、动作和阈值来自各状态对应的 JSON 文件。
int main(int argc, char **argv) {
    
    // 1. 尽早申请实时调度，随后所有控制循环都运行在该进程调度策略下。
    setProcessScheduler();

    // 2. 固定浮点日志的显示格式，便于观察控制量。
    std::cout << std::fixed << std::setprecision(3);

    // 3. 创建 IO 抽象并声明运行平台。
    //    IOSDK 当前负责 Unitree DDS 通道、低层命令发送和低层状态接收。
    IOInterface *ioInter;
    CtrlPlatform ctrlPlat;

    ioInter = new IOSDK();
    ctrlPlat = CtrlPlatform::REALROBOT;

    // 4. CtrlComponents 是各层共享的“数据总线”：
    //    它持有 LowlevelCmd、LowlevelState、IOInterface、控制周期及退出标志。
    CtrlComponents *ctrlComp = new CtrlComponents(ioInter);
    ctrlComp->ctrlPlatform = ctrlPlat;

    // 控制周期为 0.02 秒，即目标频率 50 Hz；FSM::run() 末尾负责补足周期。
    ctrlComp->dt = 0.02;
    ctrlComp->running = &running;

    // 5. ControlFrame 构造时会创建 FSM，FSM 再创建并登记所有具体状态。
    ControlFrame ctrlFrame(ctrlComp);

    // 注册 Ctrl+C 安全退出路径，避免直接中断导致堆对象来不及释放。
    signal(SIGINT, ShutDown);

    // 6. 主控制循环。每次 run() 完成一次“通信 -> 状态执行/切换 -> 周期等待”。
    while (running) {
        // FSM 内部捕获异常后会设置 exitFlag，让入口层结束循环。
        if (ctrlComp->exitFlag) break;
        ctrlFrame.run();
    }

    // 7. CtrlComponents 的析构函数会继续释放 lowCmd、lowState 和 IOSDK。
    //    ctrlFrame 是栈对象，main 返回时析构并释放其内部的 FSM。
    delete ctrlComp;
    return 0;
}
