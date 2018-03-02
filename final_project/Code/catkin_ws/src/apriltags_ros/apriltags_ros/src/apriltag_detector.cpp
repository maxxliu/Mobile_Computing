#include <apriltags_ros/apriltag_detector.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <boost/foreach.hpp>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PoseArray.h>
#include <apriltags_ros/AprilTagDetection.h>
#include <apriltags_ros/AprilTagDetectionArray.h>
#include <AprilTags/Tag16h5.h>
#include <AprilTags/Tag25h7.h>
#include <AprilTags/Tag25h9.h>
#include <AprilTags/Tag36h9.h>
#include <AprilTags/Tag36h11.h>
#include <XmlRpcException.h>

namespace apriltags_ros{

    AprilTagDetector::AprilTagDetector(ros::NodeHandle& nh, ros::NodeHandle& pnh): it_(nh){
        XmlRpc::XmlRpcValue april_tag_descriptions;
        if(!pnh.getParam("tag_descriptions", april_tag_descriptions)){
            ROS_WARN("No april tags specified");
        }
        else{
            try{
                descriptions_ = parse_tag_descriptions(april_tag_descriptions);
            } catch(XmlRpc::XmlRpcException e){
                ROS_ERROR_STREAM("Error loading tag descriptions: "<<e.getMessage());
            }
        }
        if(!pnh.getParam("sensor_frame_id", sensor_frame_id_)){
            sensor_frame_id_ = "";
        }
        if(!pnh.getParam("region", region)){
            region = "all";
        }
        if(!pnh.getParam("region_overlap", region_overlap)){
            region_overlap = 0;
        }
        if(!pnh.getParam("viewport_offset_x", viewport_offset_x)){
            viewport_offset_x = 0;
        }
        if(!pnh.getParam("viewport_offset_y", viewport_offset_y)){
            viewport_offset_y = 0;
        }

        AprilTags::TagCodes tag_codes = AprilTags::tagCodes36h11;
        tag_detector_= boost::shared_ptr<AprilTags::TagDetector>(new AprilTags::TagDetector(tag_codes));
        image_sub_ = it_.subscribeCamera("image_rect", 1, &AprilTagDetector::imageCb, this);
        image_pub_ = it_.advertise("tag_detections_image", 1);
        detections_pub_ = nh.advertise<apriltags_ros::AprilTagDetectionArray>("tag_detections", 1);
        pose_pub_ = nh.advertise<geometry_msgs::PoseArray>("tag_detections_pose", 1);
        on_switch=true;
    }
    AprilTagDetector::~AprilTagDetector(){
        image_sub_.shutdown();
    }

    void AprilTagDetector::imageCb(const sensor_msgs::ImageConstPtr& msg,const sensor_msgs::CameraInfoConstPtr& cam_info){
        cv_bridge::CvImagePtr cv_ptr;
        try{
            cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        }
        catch (cv_bridge::Exception& e){
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }

        cv::Mat gray;
        cv::cvtColor(cv_ptr->image, gray, CV_BGR2GRAY);

        double width = (double) gray.cols;
        double height = (double) gray.rows;
        double region_width = (0.5 + region_overlap) * width;
        double region_height = (0.5 + region_overlap) * height;

        double start_x, start_y;

        if( region.compare("north-west") == 0 ){
            start_x = 0.0;
            start_y = 0.0;
        }else if( region.compare("north-east") == 0 ){
            start_x = width - region_width;
            start_y = 0.0;
        }else if( region.compare("south-west") == 0 ){
            start_x = 0.0;
            start_y = height - region_height;
        }else if( region.compare("south-east") == 0 ){
            start_x = width - region_width;
            start_y = height - region_height;
        }else{
            // default
            start_x = 0.0;
            start_y = 0.0;
            region_width = width;
            region_height = height;
        }

        // crop region of interest
        cv::Mat ROI(gray, cv::Rect2f(
            start_x,
            start_y,
            region_width,
            region_height
        ));
        ROI.copyTo(gray);

        // resize image to improve performance
        // double scale_factor = 0.6;
        cv::Mat resized_img;
        // cv::resize( gray, resized_img, cv::Size(0,0), scale_factor, scale_factor );
        // gray = resized_img;


        // cv::Mat croppedImage;


        // cv::Mat gray_tmp;// = cv_ptr->image;
        // ROI.copyTo(cv_ptr->image);

        // cv::Rect roi(
        //     0,
        //     0,
        //     0.5*scale_factor*1920.0,
        //     scale_factor*1080.0
        // );
        // gray_tmp(roi);



        // cv::Mat gray = resized_img;

        // cv_ptr->image = gray;


        cv::Mat detections_image;
        cv::cvtColor(gray, detections_image, CV_GRAY2BGR);



        // cv::cvtColor(cv_ptr->image, gray, CV_BGR2GRAY);
        std::vector<AprilTags::TagDetection> detections = tag_detector_->extractTags(gray);
        ROS_DEBUG("%d tag detected", (int)detections.size());

        double fx = cam_info->K[0];
        double fy = cam_info->K[4];
        double px = cam_info->K[2];
        double py = cam_info->K[5];

        if(!sensor_frame_id_.empty())
        cv_ptr->header.frame_id = sensor_frame_id_;

        apriltags_ros::AprilTagDetectionArray tag_detection_array;
        geometry_msgs::PoseArray tag_pose_array;
        tag_pose_array.header = cv_ptr->header;

        // sensor_msgs::Image img_msg;

        double offset_x = viewport_offset_x + start_x;
        double offset_y = viewport_offset_y + start_y;

        BOOST_FOREACH(AprilTags::TagDetection detection, detections){
            std::map<int, AprilTagDescription>::const_iterator description_itr = descriptions_.find(detection.id);
            if(description_itr == descriptions_.end()){
                ROS_WARN_THROTTLE(10.0, "Found tag: %d, but no description was found for it", detection.id);
                continue;
            }
            AprilTagDescription description = description_itr->second;
            double tag_size = description.size();



            // detection.draw(cv_ptr->image);
            detection.draw(detections_image);
            Eigen::Matrix4d transform = detection.getRelativeTransform(tag_size, fx, fy, px, py, offset_x, offset_y);
            Eigen::Matrix3d rot = transform.block(0,0,3,3);
            Eigen::Quaternion<double> rot_quaternion = Eigen::Quaternion<double>(rot);

            geometry_msgs::PoseStamped tag_pose;
            tag_pose.pose.position.x = transform(0,3);
            tag_pose.pose.position.y = transform(1,3);
            tag_pose.pose.position.z = transform(2,3);
            tag_pose.pose.orientation.x = rot_quaternion.x();
            tag_pose.pose.orientation.y = rot_quaternion.y();
            tag_pose.pose.orientation.z = rot_quaternion.z();
            tag_pose.pose.orientation.w = rot_quaternion.w();
            tag_pose.header = cv_ptr->header;

            apriltags_ros::AprilTagDetection tag_detection;
            tag_detection.pose = tag_pose;
            tag_detection.id = detection.id;
            tag_detection.size = tag_size;
            tag_detection_array.detections.push_back(tag_detection);
            tag_pose_array.poses.push_back(tag_pose.pose);

            tf::Stamped<tf::Transform> tag_transform;
            tf::poseStampedMsgToTF(tag_pose, tag_transform);
            tf_pub_.sendTransform(tf::StampedTransform(tag_transform, tag_transform.stamp_, tag_transform.frame_id_, description.frame_name()));
        }

        cv_bridge::CvImage img_bridge( cv_ptr->header, sensor_msgs::image_encodings::BGR8, detections_image );

        detections_pub_.publish(tag_detection_array);
        pose_pub_.publish(tag_pose_array);
        image_pub_.publish( img_bridge.toImageMsg() );
        // image_pub_.publish(cv_ptr->toImageMsg());

        // free memory
        cv_ptr->image.release();
        gray.release();
        ROI.release();
        resized_img.release();
        detections_image.release();
    }


    std::map<int, AprilTagDescription> AprilTagDetector::parse_tag_descriptions(XmlRpc::XmlRpcValue& tag_descriptions){
        std::map<int, AprilTagDescription> descriptions;
        ROS_ASSERT(tag_descriptions.getType() == XmlRpc::XmlRpcValue::TypeArray);
        for (int32_t i = 0; i < tag_descriptions.size(); ++i) {
            XmlRpc::XmlRpcValue& tag_description = tag_descriptions[i];
            ROS_ASSERT(tag_description.getType() == XmlRpc::XmlRpcValue::TypeStruct);
            ROS_ASSERT(tag_description["id"].getType() == XmlRpc::XmlRpcValue::TypeInt);
            ROS_ASSERT(tag_description["size"].getType() == XmlRpc::XmlRpcValue::TypeDouble);

            int id = (int)tag_description["id"];
            double size = (double)tag_description["size"];

            std::string frame_name;
            if(tag_description.hasMember("frame_id")){
                ROS_ASSERT(tag_description["frame_id"].getType() == XmlRpc::XmlRpcValue::TypeString);
                frame_name = (std::string)tag_description["frame_id"];
            }
            else{
                std::stringstream frame_name_stream;
                frame_name_stream << "tag_" << id;
                frame_name = frame_name_stream.str();
            }
            AprilTagDescription description(id, size, frame_name);
            ROS_INFO_STREAM("Loaded tag config: "<<id<<", size: "<<size<<", frame_name: "<<frame_name);
            descriptions.insert(std::make_pair(id, description));
        }
        return descriptions;
    }

}
