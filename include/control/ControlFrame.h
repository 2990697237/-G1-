
#ifndef CONTROLFRAME_H
#define CONTROLFRAME_H

/*
 * ControlFrame 是 main() 与 FSM 之间的轻量门面。
 *
 * main() 只需要知道“每周期调用 run()”，无需了解具体状态；
 * 实际的通信、状态执行和状态切换全部由内部的 FSM 完成。
 */

// FSM 提供状态机控制器；CtrlComponents 是各控制层共享的上下文。
#include "FSM/FSM.h"
#include "control/CtrlComponents.h"

class ControlFrame{
public:
	// 保存共享上下文，并在实现文件中据此创建唯一的 FSM 控制器。
	ControlFrame(CtrlComponents *ctrlComp);

	// ControlFrame 拥有 _FSMController，因此析构时负责释放它；
	// _ctrlComp 的所有权仍在 main()，这里不能重复 delete。
	~ControlFrame(){
		delete _FSMController;
	}

	// 向下转发一个控制周期，调用链为 ControlFrame::run -> FSM::run。
	void run();
private:
	// 被本对象拥有的状态机实例，负责所有状态对象的生命周期。
	FSM* _FSMController;

	// 非拥有指针：指向 main() 创建的共享控制组件。
	CtrlComponents *_ctrlComp;
};

#endif  //CONTROLFRAME_H
