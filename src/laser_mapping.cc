#include <tf/transform_broadcaster.h>
#include <yaml-cpp/yaml.h>
#include <execution>
#include <fstream>

#include "laser_mapping.h"
#include "utils.h"

namespace faster_lio {

bool LaserMapping::InitROS(ros::NodeHandle &nh) {
    LoadParams(nh);
    SubAndPubToROS(nh);

    // localmap init (after LoadParams)
    ivox_ = std::make_shared<IVoxType>(ivox_options_);

    // esekf init
    std::vector<double> epsi(23, 0.001);
    kf_.init_dyn_share(
        get_f, df_dx, df_dw,
        [this](state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data) { ObsModel(s, ekfom_data); },
        options::NUM_MAX_ITERATIONS, epsi.data());

    return true;
}

bool LaserMapping::InitWithoutROS(const std::string &config_yaml) {
    LOG(INFO) << "init laser mapping from " << config_yaml;
    if (!LoadParamsFromYAML(config_yaml)) {
        return false;
    }

    // localmap init (after LoadParams)
    ivox_ = std::make_shared<IVoxType>(ivox_options_);

    // esekf init
    std::vector<double> epsi(23, 0.001);
    kf_.init_dyn_share(
        get_f, df_dx, df_dw,
        [this](state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data) { ObsModel(s, ekfom_data); },
        options::NUM_MAX_ITERATIONS, epsi.data());

    if (std::is_same<IVoxType, IVox<3, IVoxNodeType::PHC, pcl::PointXYZI>>::value == true) {
        LOG(INFO) << "using phc ivox";
    } else if (std::is_same<IVoxType, IVox<3, IVoxNodeType::DEFAULT, pcl::PointXYZI>>::value == true) {
        LOG(INFO) << "using default ivox";
    }

    return true;
}

bool LaserMapping::LoadParams(ros::NodeHandle &nh) {
    // get params from param server
    int lidar_type, ivox_nearby_type;
    double gyr_cov, acc_cov, b_gyr_cov, b_acc_cov;
    double filter_size_surf_min;
    common::V3D lidar_T_wrt_IMU;
    common::M3D lidar_R_wrt_IMU;

    nh.param<bool>("path_save_en", path_save_en_, true);
    nh.param<bool>("publish/path_publish_en", path_pub_en_, true);
    nh.param<bool>("publish/scan_publish_en", scan_pub_en_, true);
    nh.param<bool>("publish/dense_publish_en", dense_pub_en_, false);
    nh.param<bool>("publish/scan_bodyframe_pub_en", scan_body_pub_en_, true);
    nh.param<bool>("publish/scan_effect_pub_en", scan_effect_pub_en_, false);
    nh.param<std::string>("publish/tf_imu_frame", tf_imu_frame_, "body");
    nh.param<std::string>("publish/tf_world_frame", tf_world_frame_, "camera_init");

    nh.param<int>("max_iteration", options::NUM_MAX_ITERATIONS, 4);
    nh.param<float>("esti_plane_threshold", options::ESTI_PLANE_THRESHOLD, 0.1);
    nh.param<std::string>("map_file_path", map_file_path_, "");
    nh.param<bool>("common/time_sync_en", time_sync_en_, false);
    nh.param<double>("filter_size_surf", filter_size_surf_min, 0.5);
    nh.param<double>("filter_size_map", filter_size_map_min_, 0.0);
    nh.param<double>("cube_side_length", cube_len_, 200);
    nh.param<float>("mapping/det_range", det_range_, 300.f);
    nh.param<double>("mapping/gyr_cov", gyr_cov, 0.1);
    nh.param<double>("mapping/acc_cov", acc_cov, 0.1);
    nh.param<double>("mapping/b_gyr_cov", b_gyr_cov, 0.0001);
    nh.param<double>("mapping/b_acc_cov", b_acc_cov, 0.0001);
    nh.param<double>("preprocess/blind", preprocess_->Blind(), 0.01);
    nh.param<float>("preprocess/time_scale", preprocess_->TimeScale(), 1e-3);
    nh.param<int>("preprocess/lidar_type", lidar_type, 1);
    nh.param<int>("preprocess/scan_line", preprocess_->NumScans(), 16);
    nh.param<int>("point_filter_num", preprocess_->PointFilterNum(), 2);
    nh.param<bool>("feature_extract_enable", preprocess_->FeatureEnabled(), false);
    nh.param<bool>("runtime_pos_log_enable", runtime_pos_log_, true);
    nh.param<bool>("mapping/extrinsic_est_en", extrinsic_est_en_, true);
    nh.param<bool>("pcd_save/pcd_save_en", pcd_save_en_, false);
    nh.param<int>("pcd_save/interval", pcd_save_interval_, -1);
    nh.param<std::vector<double>>("mapping/extrinsic_T", extrinT_, std::vector<double>());
    nh.param<std::vector<double>>("mapping/extrinsic_R", extrinR_, std::vector<double>());

    nh.param<float>("ivox_grid_resolution", ivox_options_.resolution_, 0.2);
    nh.param<int>("ivox_nearby_type", ivox_nearby_type, 18);

#ifdef USE_CUDA
    bool enable_cuda_acceleration = true;
    nh.param<bool>("cuda/enable_acceleration", enable_cuda_acceleration, true);
    if (enable_cuda_acceleration && CudaPointCloudProcessor::IsCudaAvailable()) {
        EnableCudaAcceleration(true);
        LOG(INFO) << "CUDA acceleration enabled";
    } else {
        LOG(INFO) << "CUDA acceleration disabled or not available";
    }
#endif

    LOG(INFO) << "lidar_type " << lidar_type;
    if (lidar_type == 1) {
        preprocess_->SetLidarType(LidarType::AVIA);
        LOG(INFO) << "Using AVIA Lidar";
    } else if (lidar_type == 2) {
        preprocess_->SetLidarType(LidarType::VELO32);
        LOG(INFO) << "Using Velodyne 32 Lidar";
    } else if (lidar_type == 3) {
        preprocess_->SetLidarType(LidarType::OUST64);
        LOG(INFO) << "Using OUST 64 Lidar";
    } else {
        LOG(WARNING) << "unknown lidar_type";
        return false;
    }

    if (ivox_nearby_type == 0) {
        ivox_options_.nearby_type_ = IVoxType::NearbyType::CENTER;
    } else if (ivox_nearby_type == 6) {
        ivox_options_.nearby_type_ = IVoxType::NearbyType::NEARBY6;
    } else if (ivox_nearby_type == 18) {
        ivox_options_.nearby_type_ = IVoxType::NearbyType::NEARBY18;
    } else if (ivox_nearby_type == 26) {
        ivox_options_.nearby_type_ = IVoxType::NearbyType::NEARBY26;
    } else {
        LOG(WARNING) << "unknown ivox_nearby_type, use NEARBY18";
        ivox_options_.nearby_type_ = IVoxType::NearbyType::NEARBY18;
    }

    path_.header.stamp = ros::Time::now();
    path_.header.frame_id = tf_world_frame_;

    voxel_scan_.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);

    lidar_T_wrt_IMU = common::VecFromArray<double>(extrinT_);
    lidar_R_wrt_IMU = common::MatFromArray<double>(extrinR_);

    p_imu_->SetExtrinsic(lidar_T_wrt_IMU, lidar_R_wrt_IMU);
    p_imu_->SetGyrCov(common::V3D(gyr_cov, gyr_cov, gyr_cov));
    p_imu_->SetAccCov(common::V3D(acc_cov, acc_cov, acc_cov));
    p_imu_->SetGyrBiasCov(common::V3D(b_gyr_cov, b_gyr_cov, b_gyr_cov));
    p_imu_->SetAccBiasCov(common::V3D(b_acc_cov, b_acc_cov, b_acc_cov));
    return true;
}

bool LaserMapping::LoadParamsFromYAML(const std::string &yaml_file) {
    // get params from yaml
    int lidar_type, ivox_nearby_type;
    double gyr_cov, acc_cov, b_gyr_cov, b_acc_cov;
    double filter_size_surf_min;
    common::V3D lidar_T_wrt_IMU;
    common::M3D lidar_R_wrt_IMU;

    auto yaml = YAML::LoadFile(yaml_file);
    
#ifdef USE_CUDA
    // Try to load CUDA configuration, default to true if not present
    bool enable_cuda_acceleration = true;
#endif
    
    try {
        path_pub_en_ = yaml["publish"]["path_publish_en"].as<bool>();
        scan_pub_en_ = yaml["publish"]["scan_publish_en"].as<bool>();
        dense_pub_en_ = yaml["publish"]["dense_publish_en"].as<bool>();
        scan_body_pub_en_ = yaml["publish"]["scan_bodyframe_pub_en"].as<bool>();
        scan_effect_pub_en_ = yaml["publish"]["scan_effect_pub_en"].as<bool>();
        tf_imu_frame_ = yaml["publish"]["tf_imu_frame"].as<std::string>("body");
        tf_world_frame_ = yaml["publish"]["tf_world_frame"].as<std::string>("camera_init");
        path_save_en_ = yaml["path_save_en"].as<bool>();

        options::NUM_MAX_ITERATIONS = yaml["max_iteration"].as<int>();
        options::ESTI_PLANE_THRESHOLD = yaml["esti_plane_threshold"].as<float>();
        time_sync_en_ = yaml["common"]["time_sync_en"].as<bool>();

        filter_size_surf_min = yaml["filter_size_surf"].as<float>();
        filter_size_map_min_ = yaml["filter_size_map"].as<float>();
        cube_len_ = yaml["cube_side_length"].as<int>();
        det_range_ = yaml["mapping"]["det_range"].as<float>();
        gyr_cov = yaml["mapping"]["gyr_cov"].as<float>();
        acc_cov = yaml["mapping"]["acc_cov"].as<float>();
        b_gyr_cov = yaml["mapping"]["b_gyr_cov"].as<float>();
        b_acc_cov = yaml["mapping"]["b_acc_cov"].as<float>();
        preprocess_->Blind() = yaml["preprocess"]["blind"].as<double>();
        preprocess_->TimeScale() = yaml["preprocess"]["time_scale"].as<double>();
        lidar_type = yaml["preprocess"]["lidar_type"].as<int>();
        preprocess_->NumScans() = yaml["preprocess"]["scan_line"].as<int>();
        preprocess_->PointFilterNum() = yaml["point_filter_num"].as<int>();
        preprocess_->FeatureEnabled() = yaml["feature_extract_enable"].as<bool>();
        extrinsic_est_en_ = yaml["mapping"]["extrinsic_est_en"].as<bool>();
        pcd_save_en_ = yaml["pcd_save"]["pcd_save_en"].as<bool>();
        pcd_save_interval_ = yaml["pcd_save"]["interval"].as<int>();
        extrinT_ = yaml["mapping"]["extrinsic_T"].as<std::vector<double>>();
        extrinR_ = yaml["mapping"]["extrinsic_R"].as<std::vector<double>>();

        ivox_options_.resolution_ = yaml["ivox_grid_resolution"].as<float>();
        ivox_nearby_type = yaml["ivox_nearby_type"].as<int>();

#ifdef USE_CUDA
        if (yaml["cuda"] && yaml["cuda"]["enable_acceleration"]) {
            enable_cuda_acceleration = yaml["cuda"]["enable_acceleration"].as<bool>();
        }
#endif
    } catch (...) {
        LOG(ERROR) << "bad conversion";
        return false;
    }

#ifdef USE_CUDA
    if (enable_cuda_acceleration && CudaPointCloudProcessor::IsCudaAvailable()) {
        EnableCudaAcceleration(true);
        LOG(INFO) << "CUDA acceleration enabled from YAML config";
    } else {
        LOG(INFO) << "CUDA acceleration disabled or not available from YAML config";
    }
#endif

    LOG(INFO) << "lidar_type " << lidar_type;
    if (lidar_type == 1) {
        preprocess_->SetLidarType(LidarType::AVIA);
        LOG(INFO) << "Using AVIA Lidar";
    } else if (lidar_type == 2) {
        preprocess_->SetLidarType(LidarType::VELO32);
        LOG(INFO) << "Using Velodyne 32 Lidar";
    } else if (lidar_type == 3) {
        preprocess_->SetLidarType(LidarType::OUST64);
        LOG(INFO) << "Using OUST 64 Lidar";
    } else {
        LOG(WARNING) << "unknown lidar_type";
        return false;
    }

    if (ivox_nearby_type == 0) {
        ivox_options_.nearby_type_ = IVoxType::NearbyType::CENTER;
    } else if (ivox_nearby_type == 6) {
        ivox_options_.nearby_type_ = IVoxType::NearbyType::NEARBY6;
    } else if (ivox_nearby_type == 18) {
        ivox_options_.nearby_type_ = IVoxType::NearbyType::NEARBY18;
    } else if (ivox_nearby_type == 26) {
        ivox_options_.nearby_type_ = IVoxType::NearbyType::NEARBY26;
    } else {
        LOG(WARNING) << "unknown ivox_nearby_type, use NEARBY18";
        ivox_options_.nearby_type_ = IVoxType::NearbyType::NEARBY18;
    }

    voxel_scan_.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);

    lidar_T_wrt_IMU = common::VecFromArray<double>(extrinT_);
    lidar_R_wrt_IMU = common::MatFromArray<double>(extrinR_);

    p_imu_->SetExtrinsic(lidar_T_wrt_IMU, lidar_R_wrt_IMU);
    p_imu_->SetGyrCov(common::V3D(gyr_cov, gyr_cov, gyr_cov));
    p_imu_->SetAccCov(common::V3D(acc_cov, acc_cov, acc_cov));
    p_imu_->SetGyrBiasCov(common::V3D(b_gyr_cov, b_gyr_cov, b_gyr_cov));
    p_imu_->SetAccBiasCov(common::V3D(b_acc_cov, b_acc_cov, b_acc_cov));

    run_in_offline_ = true;
    return true;
}

#ifdef USE_CUDA
void LaserMapping::EnableCudaAcceleration(bool enable) {
    if (cuda_processor_) {
        // Enable/disable CUDA acceleration for preprocessing as well
        if (preprocess_) {
            preprocess_->EnableCudaAcceleration(enable);
        }
        LOG(INFO) << "CUDA acceleration " << (enable ? "enabled" : "disabled") << " for LaserMapping";
    } else {
        LOG(WARNING) << "CUDA processor not available, cannot enable CUDA acceleration";
    }
}

bool LaserMapping::IsCudaAccelerationEnabled() const {
    return cuda_processor_ && preprocess_ && preprocess_->IsCudaAccelerationEnabled();
}
#endif

void LaserMapping::SubAndPubToROS(ros::NodeHandle &nh) {
    // ROS subscribe initialization
    std::string lidar_topic, imu_topic;
    nh.param<std::string>("common/lid_topic", lidar_topic, "/livox/lidar");
    nh.param<std::string>("common/imu_topic", imu_topic, "/livox/imu");

    if (preprocess_->GetLidarType() == LidarType::AVIA) {
        sub_pcl_ = nh.subscribe<livox_ros_driver::CustomMsg>(
            lidar_topic, 200000, [this](const livox_ros_driver::CustomMsg::ConstPtr &msg) { LivoxPCLCallBack(msg); });
    } else {
        sub_pcl_ = nh.subscribe<sensor_msgs::PointCloud2>(
            lidar_topic, 200000, [this](const sensor_msgs::PointCloud2::ConstPtr &msg) { StandardPCLCallBack(msg); });
    }

    sub_imu_ = nh.subscribe<sensor_msgs::Imu>(imu_topic, 200000,
                                              [this](const sensor_msgs::Imu::ConstPtr &msg) { IMUCallBack(msg); });

    // PGO-related subscribers
    sub_pgo_odom_ = nh.subscribe<nav_msgs::Odometry>("/aft_pgo_odom", 100,
                                                     [this](const nav_msgs::Odometry::ConstPtr &msg) { PGOOdomCallBack(msg); });
    sub_keyframes_id_ = nh.subscribe<std_msgs::Header>("/key_frames_ids", 100,
                                                       [this](const std_msgs::Header::ConstPtr &msg) { KeyFrameIdCallBack(msg); });

    // Service servers
    srv_save_optimized_map_ = nh.advertiseService("/save_optimized_map", 
                                                 &LaserMapping::SaveOptimizedMapService, this);
    LOG(INFO) << "SaveOptimizedMap service available at /save_optimized_map";

    // ROS publisher init
    path_.header.stamp = ros::Time::now();
    path_.header.frame_id = tf_world_frame_;

    pub_laser_cloud_world_ = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered", 100000);
    pub_laser_cloud_body_ = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered_body", 100000);
    pub_laser_cloud_lidar_ = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered_lidar", 100000);  // for PGO
    pub_laser_cloud_effect_world_ = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered_effect_world", 100000);
    pub_odom_aft_mapped_ = nh.advertise<nav_msgs::Odometry>("/Odometry", 100000);
    pub_path_ = nh.advertise<nav_msgs::Path>("/path", 100000);
}

LaserMapping::LaserMapping() {
    preprocess_.reset(new PointCloudPreprocess());
    p_imu_.reset(new ImuProcess());
    
    // Initialize map compressor with default parameters
    MapCompression::CompressionParams compression_params;
    compression_params.voxel_size = 0.1f;
    compression_params.target_compression_ratio = 0.3f;
    compression_params.preserve_edges = true;
    compression_params.preserve_corners = true;
    map_compressor_.reset(new MapCompression(compression_params));
    
#ifdef USE_CUDA
    // Initialize CUDA processor
    cuda_processor_.reset(new CudaPointCloudProcessor());
    LOG(INFO) << "CUDA acceleration initialized for LaserMapping";
#endif
}

void LaserMapping::Run() {
    if (!SyncPackages()) {
        return;
    }

    /// IMU process, kf prediction, undistortion
    p_imu_->Process(measures_, kf_, scan_undistort_);
    if (scan_undistort_->empty() || (scan_undistort_ == nullptr)) {
        LOG(WARNING) << "No point, skip this scan!";
        return;
    }

    /// the first scan
    if (flg_first_scan_) {
        state_point_ = kf_.get_x();
        scan_down_world_->resize(scan_undistort_->size());
        for (int i = 0; i < scan_undistort_->size(); i++) {
            PointBodyToWorld(&scan_undistort_->points[i], &scan_down_world_->points[i]);
        }
        ivox_->AddPoints(scan_down_world_->points);
        first_lidar_time_ = measures_.lidar_bag_time_;
        flg_first_scan_ = false;
        return;
    }
    flg_EKF_inited_ = (measures_.lidar_bag_time_ - first_lidar_time_) >= options::INIT_TIME;

    /// downsample
    Timer::Evaluate(
        [&, this]() {
            voxel_scan_.setInputCloud(scan_undistort_);
            voxel_scan_.filter(*scan_down_body_);
        },
        "Downsample PointCloud");

    int cur_pts = scan_down_body_->size();
    if (cur_pts < 5) {
        LOG(WARNING) << "Too few points, skip this scan!" << scan_undistort_->size() << ", " << scan_down_body_->size();
        return;
    }
    scan_down_world_->resize(cur_pts);
    nearest_points_.resize(cur_pts);
    residuals_.resize(cur_pts, 0);
    point_selected_surf_.resize(cur_pts, true);
    plane_coef_.resize(cur_pts, common::V4F::Zero());

    // ICP and iterated Kalman filter update
    Timer::Evaluate(
        [&, this]() {
            // iterated state estimation
            double solve_H_time = 0;
            // update the observation model, will call nn and point-to-plane residual computation
            kf_.update_iterated_dyn_share_modified(options::LASER_POINT_COV, solve_H_time);
            // save the state
            state_point_ = kf_.get_x();
            euler_cur_ = SO3ToEuler(state_point_.rot);
            pos_lidar_ = state_point_.pos + state_point_.rot * state_point_.offset_T_L_I;
        },
        "IEKF Solve and Update");

    // Store pose in history for PGO corrections
    {
        std::lock_guard<std::mutex> pose_lock(pose_history_mutex_);
        geometry_msgs::PoseStamped pose_stamped;
        pose_stamped.header.stamp = ros::Time().fromSec(lidar_end_time_);
        pose_stamped.header.frame_id = tf_world_frame_;
        
        // Set pose directly instead of using template
        pose_stamped.pose.position.x = state_point_.pos(0);
        pose_stamped.pose.position.y = state_point_.pos(1);
        pose_stamped.pose.position.z = state_point_.pos(2);
        pose_stamped.pose.orientation.x = state_point_.rot.coeffs()[0];
        pose_stamped.pose.orientation.y = state_point_.rot.coeffs()[1];
        pose_stamped.pose.orientation.z = state_point_.rot.coeffs()[2];
        pose_stamped.pose.orientation.w = state_point_.rot.coeffs()[3];
        
        pose_history_.push_back(pose_stamped);
        
        // Keep only recent poses (last 1000 poses)
        const size_t max_pose_history = 1000;
        if (pose_history_.size() > max_pose_history) {
            pose_history_.erase(pose_history_.begin(), 
                               pose_history_.begin() + (pose_history_.size() - max_pose_history));
        }
    }

    // update local map
    Timer::Evaluate([&, this]() { MapIncremental(); }, "    Incremental Mapping");

    LOG(INFO) << "[ mapping ]: In num: " << scan_undistort_->points.size() << " downsamp " << cur_pts
              << " Map grid num: " << ivox_->NumValidGrids() << " effect num : " << effect_feat_num_;

    // publish or save map pcd
    if (run_in_offline_) {
        if (pcd_save_en_) {
            PublishFrameWorld();
        }
        if (path_save_en_) {
            PublishPath(pub_path_);
        }
    } else {
        if (pub_odom_aft_mapped_) {
            PublishOdometry(pub_odom_aft_mapped_);
        }
        if (path_pub_en_ || path_save_en_) {
            PublishPath(pub_path_);
        }
        if (scan_pub_en_ || pcd_save_en_) {
            PublishFrameWorld();
        }
        if (scan_pub_en_ && scan_body_pub_en_) {
            PublishFrameBody(pub_laser_cloud_body_);
            PublishFrameLidar(pub_laser_cloud_lidar_);  // for PGO - publish lidar frame
        }
        if (scan_pub_en_ && scan_effect_pub_en_) {
            PublishFrameEffectWorld(pub_laser_cloud_effect_world_);
        }
    }

    // Debug variables
    frame_num_++;
}

void LaserMapping::StandardPCLCallBack(const sensor_msgs::PointCloud2::ConstPtr &msg) {
    mtx_buffer_.lock();
    Timer::Evaluate(
        [&, this]() {
            scan_count_++;
            if (msg->header.stamp.toSec() < last_timestamp_lidar_) {
                LOG(ERROR) << "lidar loop back, clear buffer";
                lidar_buffer_.clear();
            }

            PointCloudType::Ptr ptr(new PointCloudType());
            preprocess_->Process(msg, ptr);
            lidar_buffer_.push_back(ptr);
            time_buffer_.push_back(msg->header.stamp.toSec());
            last_timestamp_lidar_ = msg->header.stamp.toSec();
        },
        "Preprocess (Standard)");
    mtx_buffer_.unlock();
}

void LaserMapping::LivoxPCLCallBack(const livox_ros_driver::CustomMsg::ConstPtr &msg) {
    mtx_buffer_.lock();
    Timer::Evaluate(
        [&, this]() {
            scan_count_++;
            if (msg->header.stamp.toSec() < last_timestamp_lidar_) {
                LOG(WARNING) << "lidar loop back, clear buffer";
                lidar_buffer_.clear();
            }

            last_timestamp_lidar_ = msg->header.stamp.toSec();

            if (!time_sync_en_ && abs(last_timestamp_imu_ - last_timestamp_lidar_) > 10.0 && !imu_buffer_.empty() &&
                !lidar_buffer_.empty()) {
                LOG(INFO) << "IMU and LiDAR not Synced, IMU time: " << last_timestamp_imu_
                          << ", lidar header time: " << last_timestamp_lidar_;
            }

            if (time_sync_en_ && !timediff_set_flg_ && abs(last_timestamp_lidar_ - last_timestamp_imu_) > 1 &&
                !imu_buffer_.empty()) {
                timediff_set_flg_ = true;
                timediff_lidar_wrt_imu_ = last_timestamp_lidar_ + 0.1 - last_timestamp_imu_;
                LOG(INFO) << "Self sync IMU and LiDAR, time diff is " << timediff_lidar_wrt_imu_;
            }

            PointCloudType::Ptr ptr(new PointCloudType());
            preprocess_->Process(msg, ptr);
            lidar_buffer_.emplace_back(ptr);
            time_buffer_.emplace_back(last_timestamp_lidar_);
        },
        "Preprocess (Livox)");

    mtx_buffer_.unlock();
}

void LaserMapping::IMUCallBack(const sensor_msgs::Imu::ConstPtr &msg_in) {
    publish_count_++;
    sensor_msgs::Imu::Ptr msg(new sensor_msgs::Imu(*msg_in));

    if (abs(timediff_lidar_wrt_imu_) > 0.1 && time_sync_en_) {
        msg->header.stamp = ros::Time().fromSec(timediff_lidar_wrt_imu_ + msg_in->header.stamp.toSec());
    }

    double timestamp = msg->header.stamp.toSec();

    mtx_buffer_.lock();
    if (timestamp < last_timestamp_imu_) {
        LOG(WARNING) << "imu loop back, clear buffer";
        imu_buffer_.clear();
    }

    last_timestamp_imu_ = timestamp;
    imu_buffer_.emplace_back(msg);
    mtx_buffer_.unlock();
}

void LaserMapping::PGOOdomCallBack(const nav_msgs::Odometry::ConstPtr &msg) {
    // Receive corrected odometry from PGO backend
    odom_aft_pgo_ = *msg;
    pgo_correction_available_ = true;
    
    // Update the corrected path
    geometry_msgs::PoseStamped pose_stamped;
    pose_stamped.header = msg->header;
    pose_stamped.pose = msg->pose.pose;
    path_updated_.poses.push_back(pose_stamped);
    path_updated_.header = msg->header;
    path_updated_.header.frame_id = "camera_init";
    
    LOG(INFO) << "Received PGO corrected odometry at time: " << msg->header.stamp.toSec();
}

void LaserMapping::KeyFrameIdCallBack(const std_msgs::Header::ConstPtr &msg) {
    // Receive keyframe IDs from PGO backend
    keyframe_ids_.push_back(msg->seq);
    LOG(INFO) << "Received keyframe ID: " << msg->seq;
}

bool LaserMapping::SyncPackages() {
    if (lidar_buffer_.empty() || imu_buffer_.empty()) {
        return false;
    }

    /*** push a lidar scan ***/
    if (!lidar_pushed_) {
        measures_.lidar_ = lidar_buffer_.front();
        measures_.lidar_bag_time_ = time_buffer_.front();

        if (measures_.lidar_->points.size() <= 1) {
            LOG(WARNING) << "Too few input point cloud!";
            lidar_end_time_ = measures_.lidar_bag_time_ + lidar_mean_scantime_;
        } else if (measures_.lidar_->points.back().curvature / double(1000) < 0.5 * lidar_mean_scantime_) {
            lidar_end_time_ = measures_.lidar_bag_time_ + lidar_mean_scantime_;
        } else {
            scan_num_++;
            lidar_end_time_ = measures_.lidar_bag_time_ + measures_.lidar_->points.back().curvature / double(1000);
            lidar_mean_scantime_ +=
                (measures_.lidar_->points.back().curvature / double(1000) - lidar_mean_scantime_) / scan_num_;
        }

        measures_.lidar_end_time_ = lidar_end_time_;
        lidar_pushed_ = true;
    }

    if (last_timestamp_imu_ < lidar_end_time_) {
        return false;
    }

    /*** push imu_ data, and pop from imu_ buffer ***/
    double imu_time = imu_buffer_.front()->header.stamp.toSec();
    measures_.imu_.clear();
    while ((!imu_buffer_.empty()) && (imu_time < lidar_end_time_)) {
        imu_time = imu_buffer_.front()->header.stamp.toSec();
        if (imu_time > lidar_end_time_) break;
        measures_.imu_.push_back(imu_buffer_.front());
        imu_buffer_.pop_front();
    }

    lidar_buffer_.pop_front();
    time_buffer_.pop_front();
    lidar_pushed_ = false;
    return true;
}

void LaserMapping::PrintState(const state_ikfom &s) {
    LOG(INFO) << "state r: " << s.rot.coeffs().transpose() << ", t: " << s.pos.transpose()
              << ", off r: " << s.offset_R_L_I.coeffs().transpose() << ", t: " << s.offset_T_L_I.transpose();
}

void LaserMapping::MapIncremental() {
    PointVector points_to_add;
    PointVector point_no_need_downsample;

    int cur_pts = scan_down_body_->size();
    points_to_add.reserve(cur_pts);
    point_no_need_downsample.reserve(cur_pts);

    std::vector<size_t> index(cur_pts);
    for (size_t i = 0; i < cur_pts; ++i) {
        index[i] = i;
    }

    std::for_each(index.begin(), index.end(), [&](const size_t &i) {
        /* transform to world frame */
        PointBodyToWorld(&(scan_down_body_->points[i]), &(scan_down_world_->points[i]));

        /* decide if need add to map */
        PointType &point_world = scan_down_world_->points[i];
        if (!nearest_points_[i].empty() && flg_EKF_inited_) {
            const PointVector &points_near = nearest_points_[i];

            Eigen::Vector3f center =
                ((point_world.getVector3fMap() / filter_size_map_min_).array().floor() + 0.5) * filter_size_map_min_;

            Eigen::Vector3f dis_2_center = points_near[0].getVector3fMap() - center;

            if (fabs(dis_2_center.x()) > 0.5 * filter_size_map_min_ &&
                fabs(dis_2_center.y()) > 0.5 * filter_size_map_min_ &&
                fabs(dis_2_center.z()) > 0.5 * filter_size_map_min_) {
                point_no_need_downsample.emplace_back(point_world);
                return;
            }

            bool need_add = true;
            float dist = common::calc_dist(point_world.getVector3fMap(), center);
            if (points_near.size() >= options::NUM_MATCH_POINTS) {
                for (int readd_i = 0; readd_i < options::NUM_MATCH_POINTS; readd_i++) {
                    if (common::calc_dist(points_near[readd_i].getVector3fMap(), center) < dist + 1e-6) {
                        need_add = false;
                        break;
                    }
                }
            }
            if (need_add) {
                points_to_add.emplace_back(point_world);
            }
        } else {
            points_to_add.emplace_back(point_world);
        }
    });

    Timer::Evaluate(
        [&, this]() {
            ivox_->AddPoints(points_to_add);
            ivox_->AddPoints(point_no_need_downsample);
        },
        "    IVox Add Points");
}

/**
 * Lidar point cloud registration
 * will be called by the eskf custom observation model
 * compute point-to-plane residual here
 * @param s kf state
 * @param ekfom_data H matrix
 */
void LaserMapping::ObsModel(state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data) {
    int cnt_pts = scan_down_body_->size();

    std::vector<size_t> index(cnt_pts);
    for (size_t i = 0; i < index.size(); ++i) {
        index[i] = i;
    }

    Timer::Evaluate(
        [&, this]() {
            auto R_wl = (s.rot * s.offset_R_L_I).cast<float>();
            auto t_wl = (s.rot * s.offset_T_L_I + s.pos).cast<float>();

#ifdef USE_CUDA
            // Try CUDA acceleration for point transformation if available
            if (cuda_processor_ && IsCudaAccelerationEnabled() && cnt_pts > 1000) {
                // Convert rotation matrix to float array for CUDA
                auto R_matrix = R_wl.toRotationMatrix();
                float rotation_matrix[9];
                float translation_vector[3];
                
                // Copy rotation matrix data
                for (int i = 0; i < 3; ++i) {
                    for (int j = 0; j < 3; ++j) {
                        rotation_matrix[i * 3 + j] = R_matrix(i, j);
                    }
                }
                
                // Copy translation vector data  
                translation_vector[0] = t_wl.x();
                translation_vector[1] = t_wl.y();
                translation_vector[2] = t_wl.z();
                
                // Use CUDA for large point clouds
                if (cuda_processor_->TransformPointCloud(scan_down_body_, scan_down_world_, 
                                                       rotation_matrix, translation_vector)) {
                    // CUDA transformation successful, continue with CPU for nearest neighbor search
                    // TODO: Add CUDA nearest neighbor search in future iteration
                    std::for_each(index.begin(), index.end(), [&](const size_t &i) {
                        PointType &point_world = scan_down_world_->points[i];
                        auto &points_near = nearest_points_[i];
                        if (ekfom_data.converge) {
                            ivox_->GetClosestPoint(point_world, points_near, options::NUM_MATCH_POINTS);
                            point_selected_surf_[i] = points_near.size() >= options::MIN_NUM_MATCH_POINTS;
                            if (point_selected_surf_[i]) {
                                point_selected_surf_[i] =
                                    common::esti_plane(plane_coef_[i], points_near, options::ESTI_PLANE_THRESHOLD);
                            }
                        }

                        if (point_selected_surf_[i]) {
                            auto temp = point_world.getVector4fMap();
                            temp[3] = 1.0;
                            float pd2 = plane_coef_[i].dot(temp);
                            
                            PointType &point_body = scan_down_body_->points[i];
                            common::V3F p_body = point_body.getVector3fMap();
                            bool valid_corr = p_body.norm() > 81 * pd2 * pd2;
                            if (valid_corr) {
                                point_selected_surf_[i] = true;
                                residuals_[i] = pd2;
                            }
                        }
                    });
                    return; // Skip CPU fallback
                }
            }
#endif

            /** closest surface search and residual computation (CPU fallback) **/
            std::for_each(index.begin(), index.end(), [&](const size_t &i) {
                PointType &point_body = scan_down_body_->points[i];
                PointType &point_world = scan_down_world_->points[i];

                /* transform to world frame */
                common::V3F p_body = point_body.getVector3fMap();
                point_world.getVector3fMap() = R_wl * p_body + t_wl;
                point_world.intensity = point_body.intensity;

                auto &points_near = nearest_points_[i];
                if (ekfom_data.converge) {
                    /** Find the closest surfaces in the map **/
                    ivox_->GetClosestPoint(point_world, points_near, options::NUM_MATCH_POINTS);
                    point_selected_surf_[i] = points_near.size() >= options::MIN_NUM_MATCH_POINTS;
                    if (point_selected_surf_[i]) {
                        point_selected_surf_[i] =
                            common::esti_plane(plane_coef_[i], points_near, options::ESTI_PLANE_THRESHOLD);
                    }
                }

                if (point_selected_surf_[i]) {
                    auto temp = point_world.getVector4fMap();
                    temp[3] = 1.0;
                    float pd2 = plane_coef_[i].dot(temp);

                    bool valid_corr = p_body.norm() > 81 * pd2 * pd2;
                    if (valid_corr) {
                        point_selected_surf_[i] = true;
                        residuals_[i] = pd2;
                    }
                }
            });
        },
        "    ObsModel (Lidar Match)");

    effect_feat_num_ = 0;

    corr_pts_.resize(cnt_pts);
    corr_norm_.resize(cnt_pts);
    for (int i = 0; i < cnt_pts; i++) {
        if (point_selected_surf_[i]) {
            corr_norm_[effect_feat_num_] = plane_coef_[i];
            corr_pts_[effect_feat_num_] = scan_down_body_->points[i].getVector4fMap();
            corr_pts_[effect_feat_num_][3] = residuals_[i];

            effect_feat_num_++;
        }
    }
    corr_pts_.resize(effect_feat_num_);
    corr_norm_.resize(effect_feat_num_);

    if (effect_feat_num_ < 1) {
        ekfom_data.valid = false;
        LOG(WARNING) << "No Effective Points!";
        return;
    }

    Timer::Evaluate(
        [&, this]() {
            /*** Computation of Measurement Jacobian matrix H and measurements vector ***/
            ekfom_data.h_x = Eigen::MatrixXd::Zero(effect_feat_num_, 12);  // 23
            ekfom_data.h.resize(effect_feat_num_);

            index.resize(effect_feat_num_);
            const common::M3F off_R = s.offset_R_L_I.toRotationMatrix().cast<float>();
            const common::V3F off_t = s.offset_T_L_I.cast<float>();
            const common::M3F Rt = s.rot.toRotationMatrix().transpose().cast<float>();            
            std::for_each(index.begin(), index.end(), [&](const size_t &i) {
                common::V3F point_this_be = corr_pts_[i].head<3>();
                common::M3F point_be_crossmat = SKEW_SYM_MATRIX(point_this_be);
                common::V3F point_this = off_R * point_this_be + off_t;
                common::M3F point_crossmat = SKEW_SYM_MATRIX(point_this);

                /*** get the normal vector of closest surface/corner ***/
                common::V3F norm_vec = corr_norm_[i].head<3>();

                /*** calculate the Measurement Jacobian matrix H ***/
                common::V3F C(Rt * norm_vec);
                common::V3F A(point_crossmat * C);

                if (extrinsic_est_en_) {
                    common::V3F B(point_be_crossmat * off_R.transpose() * C);
                    ekfom_data.h_x.block<1, 12>(i, 0) << norm_vec[0], norm_vec[1], norm_vec[2], A[0], A[1], A[2], B[0],
                        B[1], B[2], C[0], C[1], C[2];
                } else {
                    ekfom_data.h_x.block<1, 12>(i, 0) << norm_vec[0], norm_vec[1], norm_vec[2], A[0], A[1], A[2], 0.0,
                        0.0, 0.0, 0.0, 0.0, 0.0;
                }

                /*** Measurement: distance to the closest surface/corner ***/
                ekfom_data.h(i) = -corr_pts_[i][3];
            });
        },
        "    ObsModel (IEKF Build Jacobian)");
}

/////////////////////////////////////  debug save / show /////////////////////////////////////////////////////

void LaserMapping::PublishPath(const ros::Publisher pub_path) {
    SetPosestamp(msg_body_pose_);
    msg_body_pose_.header.stamp = ros::Time().fromSec(lidar_end_time_);
    msg_body_pose_.header.frame_id = tf_world_frame_;

    /*** if path is too large, the rvis will crash ***/
    path_.poses.push_back(msg_body_pose_);
    if (run_in_offline_ == false) {
        pub_path.publish(path_);
    }
}

void LaserMapping::PublishOdometry(const ros::Publisher &pub_odom_aft_mapped) {
    odom_aft_mapped_.header.frame_id = tf_world_frame_;
    odom_aft_mapped_.child_frame_id = tf_imu_frame_;
    odom_aft_mapped_.header.stamp = ros::Time().fromSec(lidar_end_time_);  // ros::Time().fromSec(lidar_end_time_);
    SetPosestamp(odom_aft_mapped_.pose);
    pub_odom_aft_mapped.publish(odom_aft_mapped_);
    auto P = kf_.get_P();
    for (int i = 0; i < 6; i++) {
        int k = i < 3 ? i + 3 : i - 3;
        odom_aft_mapped_.pose.covariance[i * 6 + 0] = P(k, 3);
        odom_aft_mapped_.pose.covariance[i * 6 + 1] = P(k, 4);
        odom_aft_mapped_.pose.covariance[i * 6 + 2] = P(k, 5);
        odom_aft_mapped_.pose.covariance[i * 6 + 3] = P(k, 0);
        odom_aft_mapped_.pose.covariance[i * 6 + 4] = P(k, 1);
        odom_aft_mapped_.pose.covariance[i * 6 + 5] = P(k, 2);
    }

    static tf::TransformBroadcaster br;
    tf::Transform transform;
    tf::Quaternion q;
    transform.setOrigin(tf::Vector3(odom_aft_mapped_.pose.pose.position.x, odom_aft_mapped_.pose.pose.position.y,
                                    odom_aft_mapped_.pose.pose.position.z));
    q.setW(odom_aft_mapped_.pose.pose.orientation.w);
    q.setX(odom_aft_mapped_.pose.pose.orientation.x);
    q.setY(odom_aft_mapped_.pose.pose.orientation.y);
    q.setZ(odom_aft_mapped_.pose.pose.orientation.z);
    transform.setRotation(q);
    br.sendTransform(tf::StampedTransform(transform, odom_aft_mapped_.header.stamp, tf_world_frame_, tf_imu_frame_));
}

void LaserMapping::PublishFrameWorld() {
    if (!(run_in_offline_ == false && scan_pub_en_) && !pcd_save_en_) {
        return;
    }

    PointCloudType::Ptr laserCloudWorld;
    if (dense_pub_en_) {
        PointCloudType::Ptr laserCloudFullRes(scan_undistort_);
        int size = laserCloudFullRes->points.size();
        laserCloudWorld.reset(new PointCloudType(size, 1));
        for (int i = 0; i < size; i++) {
            PointBodyToWorld(&laserCloudFullRes->points[i], &laserCloudWorld->points[i]);
        }
    } else {
        laserCloudWorld = scan_down_world_;
    }

    if (run_in_offline_ == false && scan_pub_en_) {
        sensor_msgs::PointCloud2 laserCloudmsg;
        pcl::toROSMsg(*laserCloudWorld, laserCloudmsg);
        laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time_);
        laserCloudmsg.header.frame_id = tf_world_frame_;
        pub_laser_cloud_world_.publish(laserCloudmsg);
        publish_count_ -= options::PUBFRAME_PERIOD;
    }

    /**************** save map ****************/
    /* 1. make sure you have enough memories
    /* 2. noted that pcd save will influence the real-time performences **/
    if (pcd_save_en_) {
        *pcl_wait_save_ += *laserCloudWorld;

        static int scan_wait_num = 0;
        scan_wait_num++;
        if (pcl_wait_save_->size() > 0 && pcd_save_interval_ > 0 && scan_wait_num >= pcd_save_interval_) {
            pcd_index_++;
            std::string all_points_dir(std::string(std::string(ROOT_DIR) + "PCD/scans_") + std::to_string(pcd_index_) +
                                       std::string(".pcd"));
            pcl::PCDWriter pcd_writer;
            LOG(INFO) << "current scan saved to /PCD/" << all_points_dir;
            pcd_writer.writeBinary(all_points_dir, *pcl_wait_save_);
            pcl_wait_save_->clear();
            scan_wait_num = 0;
        }
    }
}

void LaserMapping::PublishFrameBody(const ros::Publisher &pub_laser_cloud_body) {
    int size = scan_undistort_->points.size();
    PointCloudType::Ptr laser_cloud_imu_body(new PointCloudType(size, 1));

    for (int i = 0; i < size; i++) {
        PointBodyLidarToIMU(&scan_undistort_->points[i], &laser_cloud_imu_body->points[i]);
    }

    sensor_msgs::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*laser_cloud_imu_body, laserCloudmsg);
    laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time_);
    laserCloudmsg.header.frame_id = "body";
    pub_laser_cloud_body.publish(laserCloudmsg);
    publish_count_ -= options::PUBFRAME_PERIOD;
}

void LaserMapping::PublishFrameLidar(const ros::Publisher &pub_laser_cloud_lidar) {
    // Publish point cloud in lidar frame for PGO (ScanContext needs lidar-centric coordinates)
    sensor_msgs::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*scan_undistort_, laserCloudmsg);
    laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time_);
    laserCloudmsg.header.frame_id = "lidar";
    pub_laser_cloud_lidar.publish(laserCloudmsg);
}

void LaserMapping::PublishFrameEffectWorld(const ros::Publisher &pub_laser_cloud_effect_world) {
    int size = corr_pts_.size();
    PointCloudType::Ptr laser_cloud(new PointCloudType(size, 1));

    for (int i = 0; i < size; i++) {
        PointBodyToWorld(corr_pts_[i].head<3>(), &laser_cloud->points[i]);
    }
    sensor_msgs::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*laser_cloud, laserCloudmsg);
    laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time_);
    laserCloudmsg.header.frame_id = tf_world_frame_;
    pub_laser_cloud_effect_world.publish(laserCloudmsg);
    publish_count_ -= options::PUBFRAME_PERIOD;
}

void LaserMapping::Savetrajectory(const std::string &traj_file) {
    std::ofstream ofs;
    ofs.open(traj_file, std::ios::out);
    if (!ofs.is_open()) {
        LOG(ERROR) << "Failed to open traj_file: " << traj_file;
        return;
    }

    ofs << "#timestamp x y z q_x q_y q_z q_w" << std::endl;
    for (const auto &p : path_.poses) {
        ofs << std::fixed << std::setprecision(6) << p.header.stamp.toSec() << " " << std::setprecision(15)
            << p.pose.position.x << " " << p.pose.position.y << " " << p.pose.position.z << " " << p.pose.orientation.x
            << " " << p.pose.orientation.y << " " << p.pose.orientation.z << " " << p.pose.orientation.w << std::endl;
    }

    ofs.close();
}

///////////////////////////  private method /////////////////////////////////////////////////////////////////////
template <typename T>
void LaserMapping::SetPosestamp(T &out) {
    out.pose.position.x = state_point_.pos(0);
    out.pose.position.y = state_point_.pos(1);
    out.pose.position.z = state_point_.pos(2);
    out.pose.orientation.x = state_point_.rot.coeffs()[0];
    out.pose.orientation.y = state_point_.rot.coeffs()[1];
    out.pose.orientation.z = state_point_.rot.coeffs()[2];
    out.pose.orientation.w = state_point_.rot.coeffs()[3];
}

void LaserMapping::PointBodyToWorld(const PointType *pi, PointType *const po) {
    common::V3D p_body(pi->x, pi->y, pi->z);
    common::V3D p_global(state_point_.rot * (state_point_.offset_R_L_I * p_body + state_point_.offset_T_L_I) +
                         state_point_.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

void LaserMapping::PointBodyToWorld(const common::V3F &pi, PointType *const po) {
    common::V3D p_body(pi.x(), pi.y(), pi.z());
    common::V3D p_global(state_point_.rot * (state_point_.offset_R_L_I * p_body + state_point_.offset_T_L_I) +
                         state_point_.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = std::abs(po->z);
}

void LaserMapping::PointBodyLidarToIMU(PointType const *const pi, PointType *const po) {
    common::V3D p_body_lidar(pi->x, pi->y, pi->z);
    common::V3D p_body_imu(state_point_.offset_R_L_I * p_body_lidar + state_point_.offset_T_L_I);

    po->x = p_body_imu(0);
    po->y = p_body_imu(1);
    po->z = p_body_imu(2);
    po->intensity = pi->intensity;
}

bool LaserMapping::SaveOptimizedMapService(faster_lio::SaveOptimizedMap::Request &req,
                                         faster_lio::SaveOptimizedMap::Response &res) {
    LOG(INFO) << "SaveOptimizedMap service called";
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    try {
        // Extract global map from IVox structure
        PointCloudType::Ptr global_map(new PointCloudType);
        if (!ExtractOptimizedGlobalMap(global_map, req.apply_pgo_corrections)) {
            res.success = false;
            res.message = "Failed to extract global map from IVox structure";
            return true;
        }
        
        res.original_points = global_map->size();
        LOG(INFO) << "Extracted global map with " << res.original_points << " points";
        
        PointCloudType::Ptr compressed_map(new PointCloudType);
        
        // Apply compression if requested
        if (req.use_advanced_compression && req.voxel_size > 0.0f) {
            // Configure compression parameters
            MapCompression::CompressionParams compression_params;
            compression_params.voxel_size = req.voxel_size;
            compression_params.target_compression_ratio = req.compression_ratio;
            compression_params.preserve_edges = req.preserve_structure;
            compression_params.preserve_corners = req.preserve_structure;
            compression_params.preserve_planar_regions = req.preserve_structure;
            
            map_compressor_->SetCompressionParams(compression_params);
            
            // Apply smart compression
            auto compression_result = map_compressor_->CompressSmart(global_map, compressed_map);
            
            if (!compression_result.success) {
                LOG(WARNING) << "Advanced compression failed, falling back to basic voxel grid";
                // Fallback to basic voxel grid
                pcl::VoxelGrid<PointType> voxel_filter;
                voxel_filter.setInputCloud(global_map);
                voxel_filter.setLeafSize(req.voxel_size, req.voxel_size, req.voxel_size);
                voxel_filter.filter(*compressed_map);
                res.compression_achieved = static_cast<float>(compressed_map->size()) / res.original_points;
            } else {
                res.compression_achieved = compression_result.compression_ratio;
                LOG(INFO) << "Applied " << compression_result.compression_method 
                          << " compression in " << compression_result.processing_time_ms << " ms";
            }
        } else if (req.voxel_size > 0.0f) {
            // Basic voxel grid downsampling
            pcl::VoxelGrid<PointType> voxel_filter;
            voxel_filter.setInputCloud(global_map);
            voxel_filter.setLeafSize(req.voxel_size, req.voxel_size, req.voxel_size);
            voxel_filter.filter(*compressed_map);
            res.compression_achieved = static_cast<float>(compressed_map->size()) / res.original_points;
        } else {
            // No compression
            compressed_map = global_map;
            res.compression_achieved = 1.0f;
        }
        
        res.compressed_points = compressed_map->size();
        
        // Determine save path
        std::string save_path = req.file_path;
        if (save_path.empty()) {
            save_path = std::string(ROOT_DIR) + "PCD/optimized_map_" + 
                       std::to_string(std::time(nullptr)) + ".pcd";
        }
        
        // Ensure .pcd extension
        if (save_path.substr(save_path.length() - 4) != ".pcd") {
            save_path += ".pcd";
        }
        
        // Create directory if it doesn't exist
        std::string dir_path = save_path.substr(0, save_path.find_last_of("/"));
        std::string mkdir_cmd = "mkdir -p " + dir_path;
        int mkdir_result = system(mkdir_cmd.c_str());
        if (mkdir_result != 0) {
            LOG(WARNING) << "Failed to create directory: " << dir_path;
        }
        
        // Save the map
        pcl::PCDWriter pcd_writer;
        if (pcd_writer.writeBinary(save_path, *compressed_map) == 0) {
            res.success = true;
            res.saved_file_path = save_path;
            res.message = "Map saved successfully";
            
            // Calculate file size
            std::ifstream file(save_path, std::ifstream::ate | std::ifstream::binary);
            if (file.is_open()) {
                res.file_size_mb = static_cast<double>(file.tellg()) / (1024.0 * 1024.0);
                file.close();
            }
            
            LOG(INFO) << "Optimized map saved to: " << save_path;
            LOG(INFO) << "Original points: " << res.original_points;
            LOG(INFO) << "Compressed points: " << res.compressed_points;
            LOG(INFO) << "Compression ratio: " << res.compression_achieved;
            LOG(INFO) << "File size: " << res.file_size_mb << " MB";
            
        } else {
            res.success = false;
            res.message = "Failed to save PCD file to: " + save_path;
            LOG(ERROR) << res.message;
        }
        
    } catch (const std::exception& e) {
        res.success = false;
        res.message = "Exception occurred: " + std::string(e.what());
        LOG(ERROR) << res.message;
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    double processing_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    LOG(INFO) << "SaveOptimizedMap service completed in " << processing_time << " ms";
    
    return true;
}

bool LaserMapping::ExtractOptimizedGlobalMap(PointCloudType::Ptr& global_map, bool apply_pgo_corrections) {
    if (!ivox_ || !global_map) {
        LOG(ERROR) << "Invalid IVox structure or output cloud pointer";
        return false;
    }
    
    try {
        global_map->clear();
        
        // Extract all points from IVox structure
        // Since IVox doesn't have a direct method to extract all points,
        // we'll use the accumulated pcl_wait_save_ if available, or reconstruct from scan history
        
        if (pcd_save_en_ && pcl_wait_save_ && !pcl_wait_save_->empty()) {
            // Use accumulated point cloud if available
            *global_map = *pcl_wait_save_;
            LOG(INFO) << "Extracted global map from accumulated point cloud: " << global_map->size() << " points";
        } else {
            LOG(WARNING) << "No accumulated point cloud available, cannot extract global map";
            LOG(WARNING) << "Enable pcd_save_en in config to accumulate points for global map extraction";
            return false;
        }
        
        // Apply PGO corrections if requested and available
        if (apply_pgo_corrections && pgo_correction_available_ && !path_updated_.poses.empty()) {
            LOG(INFO) << "Applying PGO corrections to global map";
            if (!ApplyPGOCorrections(global_map, path_updated_)) {
                LOG(WARNING) << "Failed to apply PGO corrections, using uncorrected map";
            }
        }
        
        return !global_map->empty();
        
    } catch (const std::exception& e) {
        LOG(ERROR) << "Failed to extract global map: " << e.what();
        return false;
    }
}

bool LaserMapping::ApplyPGOCorrections(PointCloudType::Ptr& map, const nav_msgs::Path& corrected_path) {
    if (!map || map->empty() || corrected_path.poses.empty()) {
        LOG(ERROR) << "Invalid input for PGO correction";
        return false;
    }
    
    try {
        std::lock_guard<std::mutex> pose_lock(pose_history_mutex_);
        
        if (pose_history_.empty()) {
            LOG(WARNING) << "No pose history available for PGO correction. "
                         << "Make sure mapping has been running to accumulate pose history.";
            return false;
        }
        
        if (corrected_path.poses.empty()) {
            LOG(WARNING) << "No PGO corrections available. Check if PGO backend is running and publishing corrections.";
            return false;
        }
        
#ifdef USE_CUDA
        // Try CUDA-accelerated PGO corrections first
        if (cuda_processor_ && cuda_processor_->IsCudaAvailable()) {
            LOG(INFO) << "Applying CUDA-accelerated PGO corrections to " << map->size() << " points";
            
            // Convert pose history to geometry_msgs format for CUDA processing
            std::vector<geometry_msgs::PoseStamped> original_poses;
            original_poses.reserve(pose_history_.size());
            
            for (const auto& pose : pose_history_) {
                original_poses.push_back(pose);
            }
            
            // Generate timestamp information from point curvature field
            // (This is a fallback - ideally we'd have proper per-point timestamps)
            std::vector<float> point_timestamps;
            point_timestamps.reserve(map->size());
            
            for (const auto& point : map->points) {
                point_timestamps.push_back(point.curvature);  // Using curvature as timestamp
            }
            
            // Create output cloud
            PointCloudType::Ptr corrected_map(new PointCloudType);
            
            // Apply CUDA-accelerated PGO corrections
            bool cuda_success = cuda_processor_->ApplyPGOCorrections(
                map, corrected_map, original_poses, corrected_path.poses, point_timestamps
            );
            
            if (cuda_success && !corrected_map->empty()) {
                // Replace original map with corrected one
                *map = *corrected_map;
                
                LOG(INFO) << "Successfully applied CUDA-accelerated PGO corrections to " << map->size() << " points";
                
                // Invalidate cache since we modified the map
                std::lock_guard<std::mutex> cache_lock(map_cache_mutex_);
                map_cache_valid_ = false;
                
                return true;
            } else {
                LOG(WARNING) << "CUDA PGO correction failed, falling back to CPU implementation";
            }
        }
#endif
        
        // CPU fallback implementation  
        LOG(INFO) << "Applying CPU-based PGO corrections to " << map->size() << " points";
        
        // For proper PGO correction, we need to:
        // 1. Segment the accumulated map by scan/timestamp
        // 2. Apply the specific PGO correction for each scan's timestamp
        // 3. Transform each scan's points with its corresponding correction
        
        size_t total_corrections_applied = 0;
        size_t original_pose_count = pose_history_.size();
        size_t corrected_pose_count = corrected_path.poses.size();
        
        LOG(INFO) << "Applying per-scan PGO corrections:";
        LOG(INFO) << "  Original poses: " << original_pose_count;
        LOG(INFO) << "  Corrected poses: " << corrected_pose_count;
        
        // Since we don't have per-point timestamp information in the accumulated map,
        // we'll apply a segmented correction approach based on pose timeline
        if (original_pose_count > 0 && corrected_pose_count > 0) {
            // Calculate number of points per pose segment
            size_t points_per_segment = map->points.size() / std::min(original_pose_count, corrected_pose_count);
            if (points_per_segment == 0) points_per_segment = 1;
            
            size_t segment_count = std::min(original_pose_count, corrected_pose_count);
            LOG(INFO) << "  Points per segment: " << points_per_segment;
            LOG(INFO) << "  Total segments: " << segment_count;
            
            for (size_t segment_idx = 0; segment_idx < segment_count; ++segment_idx) {
                // Get pose indices (work backwards from most recent)
                size_t orig_idx = original_pose_count - segment_count + segment_idx;
                size_t corr_idx = corrected_pose_count - segment_count + segment_idx;
                
                if (orig_idx >= original_pose_count || corr_idx >= corrected_pose_count) {
                    continue;
                }
                
                const auto& original_pose = pose_history_[orig_idx];
                const auto& corrected_pose = corrected_path.poses[corr_idx];
                
                // Calculate transformation for this segment
                Eigen::Vector3f translation_diff(
                    corrected_pose.pose.position.x - original_pose.pose.position.x,
                    corrected_pose.pose.position.y - original_pose.pose.position.y,
                    corrected_pose.pose.position.z - original_pose.pose.position.z
                );
                
                // Calculate rotation difference
                Eigen::Quaternionf original_quat(
                    original_pose.pose.orientation.w,
                    original_pose.pose.orientation.x,
                    original_pose.pose.orientation.y,
                    original_pose.pose.orientation.z
                );
                
                Eigen::Quaternionf corrected_quat(
                    corrected_pose.pose.orientation.w,
                    corrected_pose.pose.orientation.x,
                    corrected_pose.pose.orientation.y,
                    corrected_pose.pose.orientation.z
                );
                
                Eigen::Quaternionf rotation_diff = corrected_quat * original_quat.inverse();
                Eigen::Matrix3f rotation_matrix = rotation_diff.toRotationMatrix();
                
                // Apply transformation to points in this segment
                size_t start_idx = segment_idx * points_per_segment;
                size_t end_idx = std::min(start_idx + points_per_segment, map->points.size());
                
                for (size_t pt_idx = start_idx; pt_idx < end_idx; ++pt_idx) {
                    auto& point = map->points[pt_idx];
                    
                    // Apply rotation first
                    Eigen::Vector3f pt(point.x, point.y, point.z);
                    pt = rotation_matrix * pt;
                    
                    // Then apply translation
                    point.x = pt.x() + translation_diff.x();
                    point.y = pt.y() + translation_diff.y();
                    point.z = pt.z() + translation_diff.z();
                }
                
                total_corrections_applied += (end_idx - start_idx);
                
                LOG(INFO) << "  Segment " << segment_idx << ": " << (end_idx - start_idx) 
                          << " points, translation(" << translation_diff.transpose() << ")";
            }
            
            LOG(INFO) << "Applied PGO corrections to " << total_corrections_applied 
                      << " points across " << segment_count << " segments";
            
            // Invalidate cache since we modified the map
            std::lock_guard<std::mutex> cache_lock(map_cache_mutex_);
            map_cache_valid_ = false;
        }
        
        return total_corrections_applied > 0;
        
    } catch (const std::exception& e) {
        LOG(ERROR) << "Failed to apply PGO corrections: " << e.what();
        return false;
    }
}

void LaserMapping::Finish() {
    /**************** save map ****************/
    /* 1. make sure you have enough memories
    /* 2. pcd save will largely influence the real-time performences **/
    if (pcl_wait_save_->size() > 0 && pcd_save_en_) {
        std::string file_name = std::string("scans.pcd");
        std::string all_points_dir(std::string(std::string(ROOT_DIR) + "PCD/") + file_name);
        pcl::PCDWriter pcd_writer;
        LOG(INFO) << "current scan saved to /PCD/" << file_name;
        pcd_writer.writeBinary(all_points_dir, *pcl_wait_save_);
    }

    LOG(INFO) << "finish done";
}
}  // namespace faster_lio