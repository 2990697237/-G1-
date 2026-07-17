/*
 * WBC 状态实现：用参考动作和机器人最近状态组成观测，交给 ONNX 策略网络，
 * 再把网络动作转换成 29 个关节的位置目标。
 *
 * 一次控制周期的数据流：
 *   LowlevelState + motion_data
 *       -> _observations_compute() 生成 439 维观测
 *       -> _action_compute() 执行 ONNX 推理
 *       -> _joint_q / _targetPos_rl
 *       -> LowlevelCmd 中的 q、Kp、Kd
 *
 * 状态生命周期由 FSM 调用：enter() -> 多次 run()/checkChange() -> exit()。
 */

// 标准输入输出、文件读取和基础算法。
#include <iostream>
#include "FSM/State_WBC.h"

// 参考轨迹二进制读取器。
#include "common/read_traj.h"
#include <fstream>
#include <algorithm>

// wbc.json 的解析库。
#include <nlohmann/json.hpp>

using json = nlohmann::json;

// 构造阶段完成三件事：读取配置、加载参考动作、创建 ONNX 推理会话。
// FSMState 构造函数把共享的 lowCmd/lowState 指针接入本状态，并登记状态名 WBC。
State_WBC::State_WBC(CtrlComponents *ctrlComp)
    : FSMState(ctrlComp, FSMStateName::WBC, "wbc"){

    // PROJECT_ROOT_DIR 由 CMakeLists.txt 通过 target_compile_definitions 注入，
    // 因而程序从任意工作目录启动时都能定位项目内的配置。
    std::string config_path = std::string(PROJECT_ROOT_DIR) + "/config/wbc.json";
    std::ifstream config_file(config_path);
    if (!config_file.is_open()) {
        std::cerr << "[ERROR] Failed to open config file: " << config_path << std::endl;
        throw std::runtime_error("Cannot open config file");
    }
    
    try {
        // 配置中的模型和动作路径是相对项目根目录的路径。
        json config = json::parse(config_file);
        std::string base_path = std::string(PROJECT_ROOT_DIR) + "/";
        _model_path = base_path + config["model_path"].get<std::string>();
        _folder_path = base_path + config["motion_path"].get<std::string>();

        // 安全阈值控制姿态偏差退出；三个帧下标控制动作播放区间和预设暂停点。
        _anchor_terminate_thresh = config["safe_projgravity_threshold"].get<float>();
        _start_refer_idx = config["start_idx"].get<int>();
        _pause_refer_idx = config["pause_idx"].get<int>();
        _end_refer_idx = config["end_idx"].get<int>();
        std::cout << "[Config] Model path: " << _model_path << std::endl;
        std::cout << "[Config] Folder path: " << _folder_path << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Failed to parse config file: " << e.what() << std::endl;
        throw;
    }
    config_file.close();


    // 一次性读入动作目录中的 7 组数组。每组同时保存扁平数据和原始 shape，
    // 后续通过 frame/link/dof 下标计算扁平数组偏移。
    _bin_data_loaded = BinaryArrayReader::readBinFilesFromFolder(
        _folder_path,
        _body_ang_vel_w, _body_ang_vel_w_shape,
        _body_lin_vel_w, _body_lin_vel_w_shape,
        _body_pos_w, _body_pos_w_shape,
        _body_quat_w, _body_quat_w_shape,
        _fps, _fps_shape,
        _joint_pos, _joint_pos_shape,
        _joint_vel, _joint_vel_shape
    );

    // joint_pos 的第 0 维就是参考动作总帧数。
    _motion_frame_count = _joint_pos_shape[0];
    if (_bin_data_loaded) {
        std::cout << "[SUCCESS] Loaded motion data from: " << _folder_path << std::endl;
        std::cout << "Total motion frames: " << _motion_frame_count << std::endl;
        
    } else {
        std::cerr << "[ERROR] Failed to load some binary data!" << std::endl;
    }

    // 轨迹准备完成后加载策略，模型输入/输出维度也在这里读取。
    _loadPolicy();
}

// 用当前观测填满 4 帧历史缓冲区，避免状态刚进入时历史全为 0。
// 当前 enter() 中对此函数的调用被注释，因此实际初始历史仍是 0，
// 随着 run() 执行才逐帧被真实状态替换。
void State_WBC::_init_buffers()
{
    _last_refer_idx = 0;
    for (int i = 0; i < this->_actor_state_history_length; ++i)
    {
        _observations_compute(); 
    }
}

// 创建 ONNX Runtime 会话，并从模型元数据中读取第一个输入和输出的形状。
void State_WBC::_loadPolicy() 
{
    // 可用执行后端列表当前只被查询、未被选择；会话使用 Runtime 默认执行后端。
    auto available_providers = Ort::GetAvailableProviders();

    // 启用全部图优化，再用 wbc.json 指定的 .onnx 文件创建会话。
    _session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    _session = std::make_unique<Ort::Session>(_env, _model_path.c_str(), _session_options);

    // 项目约定模型形状为 [batch, features]，所以索引 1 是特征维度。
    Ort::TypeInfo input_type = _session->GetInputTypeInfo(0);
    auto input_shapes = input_type.GetTensorTypeAndShapeInfo().GetShape();
    Ort::TypeInfo output_type = _session->GetOutputTypeInfo(0);
    auto output_shapes = output_type.GetTensorTypeAndShapeInfo().GetShape();

    _obs_size_ = input_shapes[1];
    _action_size_ = output_shapes[1];

    // 动作缓冲区既保存本周期网络输出，也作为“上一动作”进入下一周期观测。
    _action = std::vector<float>(_action_size_, 0.0f);

}

// 生成策略网络观测。最终布局为：
//   [67 维参考动作] + [4 × 93 维机器人状态历史] = 439 维。
void State_WBC::_observations_compute()
{   
    // ---------- A. 当前机器人状态：姿态、关节状态和上一动作 ----------

    // Unitree IMU 四元数按 [w, x, y, z] 排列。
    std::vector<float> base_quat = std::vector<float>(4, 0.0f); 
    base_quat = {
        _lowState->imu.quaternion[0],  // w
        _lowState->imu.quaternion[1],  // x
        _lowState->imu.quaternion[2],  // y
        _lowState->imu.quaternion[3]}; // z

    // 把世界系重力方向旋转到机身坐标系。
    // 它比欧拉角更适合直接表达机器人倾斜方向，也是后续安全判断的依据。
    std::vector<float> projected_gravity(3);  
    projected_gravity = QuatRotateInverse(base_quat, this->_gravity_vec);
    auto obs_projected_gravity = projected_gravity;

    // 读取腰部 yaw/roll/pitch 三个电机角，并构造腰部相对机身的旋转。
    std::vector<float> waist_yrp(3);
    for(int i=0; i<3; i++)
    {
        waist_yrp[i] = _lowState->motorState[_waist_yrp_idx[i]].q;
    }
    
    Eigen::Matrix3f R_b2w = rotz(waist_yrp[0]) * rotx(waist_yrp[1]) * roty(waist_yrp[2]);
    Eigen::Matrix3f R_base = matrix_from_quat(base_quat);
    Eigen::Matrix3f R_waist = R_base * R_b2w;

    // waist_quat 当前计算后没有继续拼入观测，可把它视为为腰部姿态特征预留的中间量。
    std::vector<float> waist_quat = quat_from_matrix(R_waist);

    // dof_mapping 把“策略训练时的关节顺序”映射到“电机数组顺序”。
    // 位置先减默认站姿，使策略看到的是相对默认姿态的偏移。
    std::vector<float> dof_pos_vec;
    dof_pos_vec.reserve(NUM_DOF);
    for (int i = 0; i < NUM_DOF; ++i) {
        dof_pos_vec.push_back(_lowState->motorState[dof_mapping[i]].q - this->_default_dof_pos[dof_mapping[i]]);
    }
 
    // 关节速度同样按策略顺序组织，但不需要减默认值。
    std::vector<float> dof_vel_vec;
    dof_vel_vec.reserve(NUM_DOF);
    for (int i = 0; i < NUM_DOF; ++i) {
        dof_vel_vec.push_back(_lowState->motorState[dof_mapping[i]].dq);
    }

    // IMU 陀螺仪给出机身角速度。
    auto body_ang_vel = std::vector<float>({
        static_cast<float>(_lowState->imu.gyroscope[0]),
        static_cast<float>(_lowState->imu.gyroscope[1]),
        static_cast<float>(_lowState->imu.gyroscope[2])
    });

    // 使用与训练阶段一致的缩放系数。当前系数均为 1，因此数值不变；
    // 前两轴使用 scale_lin_vel 是现有实现，阅读时不要把它误认为线速度输入。
    body_ang_vel[0] = body_ang_vel[0] * scale_lin_vel; 
    body_ang_vel[1] = body_ang_vel[1] * scale_lin_vel;  
    body_ang_vel[2] = body_ang_vel[2] * scale_ang_vel;  
    for(int i=0; i<NUM_DOF; i++)
    {
        dof_pos_vec[i] = dof_pos_vec[i] * scale_dof_pos;  
        dof_vel_vec[i] = dof_vel_vec[i] * scale_dof_vel;  
    }

    // 单帧机器人状态共 93 维：
    // 3 角速度 + 3 投影重力 + 29 关节位置 + 29 关节速度 + 29 上一动作。
    std::vector<float> current_robot_state;
    current_robot_state.reserve(3 + 3 + dof_pos_vec.size() + dof_vel_vec.size() + _action.size());
    current_robot_state.insert(current_robot_state.end(), body_ang_vel.begin(), body_ang_vel.end());
    current_robot_state.insert(current_robot_state.end(), obs_projected_gravity.begin(), obs_projected_gravity.end());
    current_robot_state.insert(current_robot_state.end(), dof_pos_vec.begin(), dof_pos_vec.end());
    current_robot_state.insert(current_robot_state.end(), dof_vel_vec.begin(), dof_vel_vec.end());
    current_robot_state.insert(current_robot_state.end(), _action.begin(), _action.end());
    
    // 维护固定长度的 4 帧 FIFO：删除最老的 93 维，把本帧追加到末尾。
    _robot_state_obs_buf.erase(_robot_state_obs_buf.begin(),
                                _robot_state_obs_buf.begin() + _robot_state_dim);
    _robot_state_obs_buf.insert(_robot_state_obs_buf.end(),
                                current_robot_state.begin(),
                                current_robot_state.end());

    // ---------- B. 参考动作：从扁平二进制数组按帧取数据 ----------

    // 参考数据以连续一维 vector 保存；以下 lambda 封装 shape 到线性偏移的换算。
    auto get_pos = [this](int frame_idx, int link_idx) -> std::vector<float>
    {
        int num_links = _body_pos_w_shape[1];
        int base = frame_idx * num_links * 3 + link_idx * 3;
        return {_body_pos_w[base], _body_pos_w[base + 1], _body_pos_w[base + 2]};
    };
    auto get_quat = [this](int frame_idx, int link_idx) -> std::vector<float>
    {
        int num_links = _body_quat_w_shape[1];
        int base = frame_idx * num_links * 4 + link_idx * 4;
        return {_body_quat_w[base], _body_quat_w[base + 1], _body_quat_w[base + 2], _body_quat_w[base + 3]};
    };
    auto get_dof_pos = [this](int frame_idx) -> std::vector<float>
    {
        int num_dofs = _joint_pos_shape[1];
        int base = frame_idx * num_dofs;
        std::vector<float> dof_pos(num_dofs);
        for (int i = 0; i < num_dofs; ++i)
        {
            dof_pos[i] = _joint_pos[base + i];
        }
        return dof_pos;
    };
    auto get_dof_vel = [this](int frame_idx) -> std::vector<float>
    {
        int num_dofs = _joint_vel_shape[1];
        int base = frame_idx * num_dofs;
        std::vector<float> dof_vel(num_dofs);
        for (int i = 0; i < num_dofs; ++i)
        {
            dof_vel[i] = _joint_vel[base + i];
        }
        return dof_vel;
    };

    // 这些变量保存当前/上一参考帧以及坐标系对齐过程中的中间结果。
    std::vector<float> cur_refer_anchor_pos;
    std::vector<float> cur_refer_dof_pos;
    std::vector<float> cur_refer_dof_vel;
    std::vector<float> cur_refer_anchor_quat;
    std::vector<float> ref_yaw_quat;
    std::vector<float> ref_yaw_quat_conj;
    std::vector<float> yaw_quat_delta;
    std::vector<float> aligned_cur_refer_anchor_quat;
    std::vector<float> motion_projected_gravity;
    std::vector<float> base_yaw_quat = yaw_quat(base_quat);
    std::vector<float> last_refer_anchor_pos;
    std::vector<float> last_refer_anchor_quat;

    // 四类特征分别展平；当 predictive_horizon > 1 时会按预测时刻依次追加。
    std::vector<float> tgt_anchor_ori_b_flat;              
    std::vector<float> tgt_anchor_pos_b_flat; 
    std::vector<float> tgt_dof_pos_flat; 
    std::vector<float> tgt_dof_vel_flat; 

    for (int i = 0; i < _mimic_obs_predictive_horizon; i++)
    {
        // 预测帧以 _frame_interval 为间隔。当前 horizon=1，所以只使用当前参考帧；
        // last_idx 用来计算相邻参考帧之间的锚点相对运动。
        int idx = _refer_idx + i * _frame_interval;
        int last_idx = idx - _frame_interval;

        // 暂停时当前帧和上一帧都固定，参考关节速度也在下方强制为 0。
        if(_pause_flag)
        {
            idx = _refer_idx;
            last_idx = _refer_idx;
        }
        if (idx >= _end_refer_idx)
            idx = _end_refer_idx;
        else if (idx <= 1)
            idx = 1;
        if (last_idx >= _end_refer_idx)
            last_idx = _end_refer_idx;
        else if (last_idx <= 1)
            last_idx = 1;

        // 读取根锚点的位置、姿态，以及 29 个参考关节的位置和速度。
        cur_refer_anchor_pos = get_pos(idx, _anchor_idx);
        cur_refer_dof_pos = get_dof_pos(idx);
        if(_pause_flag)
        {
            cur_refer_dof_vel = std::vector<float>(NUM_DOF, 0.0f);
        }
        else
        {
            cur_refer_dof_vel = get_dof_vel(idx);
        }
        
        cur_refer_anchor_quat = get_quat(idx, _anchor_idx);

        // 只对齐机器人与参考动作的偏航角，避免世界坐标中的初始朝向差异
        // 被误认为动作模仿误差；对齐姿态当前用于计算参考投影重力。
        ref_yaw_quat = yaw_quat(cur_refer_anchor_quat);
        ref_yaw_quat_conj = quat_conjugate(ref_yaw_quat);
        yaw_quat_delta = quat_multiply(base_yaw_quat, ref_yaw_quat_conj);
        aligned_cur_refer_anchor_quat = quat_multiply(yaw_quat_delta, cur_refer_anchor_quat);

        last_refer_anchor_pos = get_pos(last_idx, _anchor_idx);
        last_refer_anchor_quat = get_quat(last_idx, _anchor_idx);

        // 安全判断只需当前预测窗口的第一帧。
        if (i == 0)
            motion_projected_gravity = quat_apply_inverse(aligned_cur_refer_anchor_quat, _gravity_vec);

        // 求参考锚点从 last_idx 到 idx 的相对位姿，消除绝对世界位置的影响。
        auto [cur_target_pos, cur_target_quat] = subtract_frame_transforms(
            last_refer_anchor_pos,
            last_refer_anchor_quat,
            &cur_refer_anchor_pos,
            &cur_refer_anchor_quat
        );

        
        // 将相对四元数转为旋转矩阵，并取前两列形成连续的 6D 姿态表示。
        // 相比直接回归四元数，这种表示没有 q 与 -q 表示同一姿态的二义性。
        Eigen::Matrix3f mat = matrix_from_quat(cur_target_quat);
        std::vector<float> tgt_anchor_ori_b(6);
        tgt_anchor_ori_b[0] = mat(0, 0);
        tgt_anchor_ori_b[1] = mat(0, 1);
        tgt_anchor_ori_b[2] = mat(1, 0);
        tgt_anchor_ori_b[3] = mat(1, 1);
        tgt_anchor_ori_b[4] = mat(2, 0);
        tgt_anchor_ori_b[5] = mat(2, 1);

        // 每个预测时刻贡献 29 + 29 + 3 + 6 = 67 维参考特征。
        tgt_dof_pos_flat.insert(tgt_dof_pos_flat.end(), cur_refer_dof_pos.begin(), cur_refer_dof_pos.end());
        tgt_dof_vel_flat.insert(tgt_dof_vel_flat.end(), cur_refer_dof_vel.begin(), cur_refer_dof_vel.end());
        tgt_anchor_pos_b_flat.insert(tgt_anchor_pos_b_flat.end(), cur_target_pos.begin(), cur_target_pos.end());
        tgt_anchor_ori_b_flat.insert(tgt_anchor_ori_b_flat.end(), tgt_anchor_ori_b.begin(), tgt_anchor_ori_b.end());
    }

    // 参考观测的拼接顺序必须与模型训练时完全相同。
    std::vector<float> mimic_obs;
    mimic_obs.insert(mimic_obs.end(), tgt_dof_pos_flat.begin(), tgt_dof_pos_flat.end());
    mimic_obs.insert(mimic_obs.end(), tgt_dof_vel_flat.begin(), tgt_dof_vel_flat.end());
    mimic_obs.insert(mimic_obs.end(), tgt_anchor_pos_b_flat.begin(), tgt_anchor_pos_b_flat.end());
    mimic_obs.insert(mimic_obs.end(), tgt_anchor_ori_b_flat.begin(), tgt_anchor_ori_b_flat.end());

    // ---------- C. 安全检查与最终拼接 ----------

    // 比较参考姿态与真实姿态在竖直重力分量上的偏差。
    // 超过配置阈值只设置标志，真正切到 PASSIVE 由 checkChange() 完成。
    float anchor_proj_gravity_error = std::abs(motion_projected_gravity[2] - projected_gravity[2]);
    // std::cout << "anchor_proj_gravity_error: " << anchor_proj_gravity_error << std::endl;
    if (anchor_proj_gravity_error > _anchor_terminate_thresh)
    {
        _terminate_flag = true;
        std::cout << "current _anchor_terminate_thresh: " << _anchor_terminate_thresh << std::endl;
        std::cout << "[Warning] Large anchor projected gravity error: " << anchor_proj_gravity_error << std::endl;
    }

    // 最终顺序是参考观测在前、机器人历史在后；该顺序同样属于模型接口契约。
    this->_observation.clear();
    this->_observation = std::vector<float>();
    this->_observation.reserve(
        _robot_state_obs_buf.size() + mimic_obs.size());
    this->_observation.insert(this->_observation.end(), mimic_obs.begin(), mimic_obs.end());
    this->_observation.insert(this->_observation.end(), _robot_state_obs_buf.begin(), _robot_state_obs_buf.end());
    
    // 逐元素限幅，防止异常传感器值把网络输入推到训练分布之外。
    for(int i = 0; i < this->_observation.size(); i++)
    {
        this->_observation[i] = std::max(-clip_observations, std::min(this->_observation[i], clip_observations));
    }

}

// 将 439 维观测包装成 ONNX Tensor，执行推理并转换为电机顺序的关节目标。
void State_WBC::_action_compute()
{
    try
    {
        // 观测 vector 位于 CPU 内存，OrtArenaAllocator 负责 Tensor 的内存描述。
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeCPU);

        std::vector<Ort::Value> input_tensors;

        // 输入 batch 固定为 1；特征数按本类常量计算为 93*4 + 67*1 = 439。
        // 它必须与 _loadPolicy() 读出的模型输入维度一致。
        std::vector<int64_t> obs_shape = {1,
            _robot_state_dim * _actor_state_history_length 
            + _reference_dim * _mimic_obs_predictive_horizon
        }; 

        input_tensors.push_back(Ort::Value::CreateTensor<float>(
            memory_info,
            _observation.data(),
            _observation.size(),
            obs_shape.data(),
            obs_shape.size()));

        // 输入/输出节点名在头文件中固定为 "obs" 和 "actions"。
        auto output_tensors = _session->Run(
            Ort::RunOptions{nullptr},
            _input_names.data(),
            input_tensors.data(),
            input_tensors.size(),
            _output_names.data(),
            1
        );

        // 把 Runtime 所拥有的输出 Tensor 复制到本状态的持久动作缓冲区。
        float *actions = output_tensors[0].GetTensorMutableData<float>();
        std::memcpy(_action.data(), actions, _action.size() * sizeof(float));

        // 策略输出是相对默认姿态的归一化偏移：
        // 先限幅，再乘 action_scale，最后加对应电机的默认关节角。
        std::vector<float> actions_scaled(_action.size());
        for (int i = 0; i < _action.size(); i++)
        {
            _action[i] = std::max(-clip_actions, std::min(_action[i], clip_actions));
            actions_scaled[i] = _action[i] * action_scale + _default_dof_pos[dof_mapping[i]]; // action_scale
            this->_joint_q[dof_mapping[i]] = actions_scaled[i];
        }
    }
    // 推理异常在本层记录但不向外抛出；本周期会继续使用 _joint_q 中已有的目标值。
    catch (const Ort::Exception &e)
    {
        std::cerr << "ONNX Runtime error: " << e.what() << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Standard exception: " << e.what() << std::endl;
    }
    catch (...)
    {
        std::cerr << "Unknown error occurred" << std::endl;
    }
}

// 每次从其他状态切入 WBC 时执行：重置动作播放进度和低层 PD 命令。
void State_WBC::enter()
{
    // 默认从配置起始帧继续播放，不处于暂停或安全终止状态。
    _pause_flag = false;
    _terminate_flag = false;
    _pause_curr_flag = false;
    _refer_idx = _start_refer_idx;
    _last_refer_idx = _refer_idx;

    // 负的暂停帧无意义，回退到第 0 帧。
    if(_pause_refer_idx<0)
    {
        _pause_refer_idx = 0;
    }

    // end_idx=-1 表示播放到动作末尾；小于 start_idx 时也采用同一回退策略。
    if (_end_refer_idx < 0 || _end_refer_idx < _start_refer_idx)
    {
        if (_end_refer_idx < _start_refer_idx)
        {
            std::cout << "[WARNING]: end_idx is smaller than start_idx, defaulting to the length of the motion." << std::endl;
        }
        _end_refer_idx = _motion_frame_count - 1;
    } 

    // 为全部 29 个电机建立位置 PD 命令：
    // 初始目标取当前实测角，避免切入瞬间直接跳到默认姿态。
    for (int i = 0; i < NUM_DOF; i++)
    {
        _lowCmd->motorCmd[i].mode = 10;
        _lowCmd->motorCmd[i].q = _lowState->motorState[i].q;  
        _lowCmd->motorCmd[i].dq = 0;
        _lowCmd->motorCmd[i].tau = 0;
        _lowCmd->motorCmd[i].Kp = this->dof_Kps[i];
        _lowCmd->motorCmd[i].Kd = this->dof_Kds[i];
        this->_targetPos_rl[i] = this->_default_dof_pos[i];  
        this->_last_targetPos_rl[i] = _lowState->motorState[i].q;
        this->_joint_q[i] = this->_default_dof_pos[i];
    }

    // 若启用该调用，会用重复的当前状态预热 4 帧历史缓冲；当前实现保持关闭。
    // _init_buffers();
}

// WBC 状态的单周期主体：推进参考帧、推理、写入 29 路低层命令。
void State_WBC::run()
{
    // 正常播放每周期前进一帧；暂停则保持预设帧或按键发生时的当前帧。
    if(!_pause_flag){
        _refer_idx++;
    }
    else{
        if (!_pause_curr_flag)
            _refer_idx = _pause_refer_idx;
    }

    // 到达末帧后钳住索引，不自动离开 WBC。
    if (_refer_idx >= _end_refer_idx)
    {
        _refer_idx = _end_refer_idx;
    }
    _observations_compute(); 
    _action_compute(); 

    // 将按电机顺序排列的推理结果复制为本周期目标。
    memcpy(this->_targetPos_rl, this->_joint_q, sizeof(this->_joint_q));
   
    // 对每个电机写入位置 PD 控制命令：目标速度/前馈力矩为 0，
    // 关节位置来自策略，Kp/Kd 来自 State_WBC.h 的电机参数表。
    for (int j = 0; j < NUM_DOF; j++) 
    {
        _lowCmd->motorCmd[j].mode = 10;
        _lowCmd->motorCmd[j].q = _targetPos_rl[j];
        _lowCmd->motorCmd[j].dq = 0;
        _lowCmd->motorCmd[j].tau = 0;
        _lowCmd->motorCmd[j].Kp = this->dof_Kps[j];
        _lowCmd->motorCmd[j].Kd = this->dof_Kds[j];
        this->_last_targetPos_rl[j] = _targetPos_rl[j];
    }

    // 保存进度并在同一终端行刷新运行状态。
    _last_refer_idx = _refer_idx;
    std::string pause_string = _pause_flag ? " | Press R1 to resume..." : " | Press R2 to pause...";
    std::cout << "\r[State_WBC] Running WBC state. Refer idx: " << _refer_idx << "/" << _end_refer_idx << pause_string << std::flush;
}

// 离开状态时当前只打印日志；具体下一状态会在随后 enter() 中重设命令。
void State_WBC::exit()
{
    std::cout << "[State_WBC] Exiting WBC state." << std::endl;
}

// 根据用户命令和安全标志决定下一状态。
// 返回 WBC 表示不切换；FSM 只有看到不同的状态名才进入 CHANGE 阶段。
FSMStateName State_WBC::checkChange()
{
    // L2+B 或姿态偏差超阈值：优先回到被动保护状态。
    if (_lowState->userCmd == UserCommand::L2_B)
    {
        return FSMStateName::PASSIVE;
    }
    else if (_terminate_flag)
    {
        return FSMStateName::PASSIVE;
    }

    // R2+A：切换到 AMP 策略状态。
    else if(_lowState->userCmd == UserCommand::R2_A){ 
        return FSMStateName::AMP;
    }

    // R2：暂停并跳到配置中的 pause_idx。
    else if (_lowState->userCmd == UserCommand::R2 && !_pause_flag)
    { 
        _pause_flag = true;
        std::cout << std::endl <<"WBC Pause" <<std::endl;
        return FSMStateName::WBC;
    }

    // L2：暂停在按键发生时的当前参考帧。
    else if (_lowState->userCmd == UserCommand::L2 && !_pause_flag)
    {
        _pause_flag = true;
        _pause_curr_flag = true;
        std::cout << std::endl
                  << "WBC Pause" << std::endl;
        return FSMStateName::WBC;
    }

    // R1：解除任一种暂停方式，从当前 _refer_idx 继续播放。
    else if (_lowState->userCmd == UserCommand::R1 && _pause_flag)
    { 
        _pause_flag = false;
        _pause_curr_flag = false;
        std::cout << std::endl << "WBC Resume" << std::endl;
        return FSMStateName::WBC;
    }

    // SELECT 通过异常交给 FSM::run() 的 catch 设置 exitFlag，最终结束 main 循环。
    else if(_lowState->userCmd == UserCommand::SELECT){
        throw std::runtime_error("exit..");
        return FSMStateName::PASSIVE;
    }

    // 没有切换条件时保持 WBC。
    else{ 
        return FSMStateName::WBC;
    }
}
