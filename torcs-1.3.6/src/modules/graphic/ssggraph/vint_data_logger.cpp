#include <opencv2/opencv.hpp>
#include <GL/gl.h>
#include <GL/glut.h>
#include <car.h>
#include <fstream>
#include <sstream>
#include <sys/stat.h>
#include <sys/time.h>
#include <errno.h>

// Global logger state
static bool g_logging_enabled = false;
static std::string g_log_dir;
static std::ofstream g_pose_log;
static std::ofstream g_goal_log;
static uint32_t g_frame_counter = 0;
static uint64_t g_session_start_time = 0;

// ViNT Image Configuration
static const int VINT_IMAGE_WIDTH = 512;   // Increased from 224 for better detail
static const int VINT_IMAGE_HEIGHT = 512;  // Increased from 224 for better detail

// Create directory recursively (like mkdir -p)
bool createDirectory(const std::string& path) {
    struct stat st = {0};
    if (stat(path.c_str(), &st) == -1) {
        // Try to create parent directory first
        size_t pos = path.find_last_of('/');
        if (pos != std::string::npos && pos > 0) {
            std::string parent = path.substr(0, pos);
            if (!createDirectory(parent)) {
                printf("Failed to create parent directory: %s\n", parent.c_str());
                return false;
            }
        }
        
        if (mkdir(path.c_str(), 0700) != 0) {
            printf("mkdir failed for: %s (errno: %d)\n", path.c_str(), errno);
            return false;
        }
        printf("Created directory: %s\n", path.c_str());
    }
    return true;
}

// Get current timestamp in microseconds
uint64_t getCurrentTimestamp() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return tv.tv_sec * 1000000ULL + tv.tv_usec;
}

// Initialize data logger
extern "C" int initVintDataLogger(const char* track_name) {
    // Create session ID (timestamp)
    g_session_start_time = getCurrentTimestamp();
    std::stringstream ss;
    ss << "./vint_torcs_logs/" << track_name << "_" << g_session_start_time;
    g_log_dir = ss.str();
    
    // Create directory structure
    if (!createDirectory(g_log_dir)) {
        printf("Failed to create log directory: %s\n", g_log_dir.c_str());
        return 0;
    }
    
    std::string frames_dir = g_log_dir + "/frames";
    if (!createDirectory(frames_dir)) {
        printf("Failed to create frames directory: %s\n", frames_dir.c_str());
        return 0;
    }
    
    // Open pose log
    std::string pose_file = g_log_dir + "/pose.csv";
    g_pose_log.open(pose_file);
    if (!g_pose_log.is_open()) {
        printf("Failed to open pose log: %s\n", pose_file.c_str());
        return 0;
    }
    g_pose_log << "timestamp,frame_id,x,y,theta,speed\n";
    
    // Open goal log
    std::string goal_file = g_log_dir + "/goal.csv";
    g_goal_log.open(goal_file);
    if (!g_goal_log.is_open()) {
        printf("Failed to open goal log: %s\n", goal_file.c_str());
        return 0;
    }
    g_goal_log << "timestamp,frame_id,goal_x,goal_y\n";
    
    // Create metadata file
    std::string metadata_file = g_log_dir + "/metadata.json";
    std::ofstream metadata_log(metadata_file);
    if (metadata_log.is_open()) {
        metadata_log << "{\n";
        metadata_log << "  \"track_name\": \"" << track_name << "\",\n";
        metadata_log << "  \"session_id\": " << g_session_start_time << ",\n";
        metadata_log << "  \"frame_resolution\": [" << VINT_IMAGE_WIDTH << ", " << VINT_IMAGE_HEIGHT << "],\n";
        metadata_log << "  \"camera_type\": \"ego_centric_cockpit\",\n";
        metadata_log << "  \"lookahead_distance\": 10.0,\n";
        metadata_log << "  \"start_time\": " << g_session_start_time << "\n";
        metadata_log << "}\n";
        metadata_log.close();
    }
    
    g_logging_enabled = true;
    g_frame_counter = 0;
    
    printf("ViNT data logger initialized: %s\n", g_log_dir.c_str());
    printf("Capturing %dx%d ego-centric frames\n", VINT_IMAGE_WIDTH, VINT_IMAGE_HEIGHT);
    return 1;
}

// Cleanup data logger
extern "C" void cleanupVintDataLogger() {
    if (g_pose_log.is_open()) {
        g_pose_log.close();
    }
    if (g_goal_log.is_open()) {
        g_goal_log.close();
    }
    g_logging_enabled = false;
    printf("ViNT data logger cleaned up\n");
}

// Calculate goal point (10m ahead on centerline)
void calculateGoalPoint(tCarElt* car, float& goal_x, float& goal_y) {
    // Simple goal: 10m ahead on current heading
    float lookahead_distance = 10.0f;
    goal_x = car->pub.DynGCg.pos.x + lookahead_distance * cos(car->pub.DynGCg.pos.az);
    goal_y = car->pub.DynGCg.pos.y + lookahead_distance * sin(car->pub.DynGCg.pos.az);
}

// Capture and save frame
void saveFrame(int scrx, int scry, int scrw, int scrh) {
    if (!g_logging_enabled) return;
    
    // Capture OpenGL framebuffer
    glReadBuffer(GL_FRONT);
    glPixelStorei(GL_PACK_ROW_LENGTH, 0);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    
    // Read full resolution frame
    unsigned char* full_buffer = new unsigned char[scrw * scrh * 3];
    glReadPixels(scrx, scry, scrw, scrh, GL_RGB, GL_UNSIGNED_BYTE, full_buffer);
    
    // Convert to OpenCV Mat (note: OpenGL origin is bottom-left, OpenCV is top-left)
    cv::Mat full_frame(scrh, scrw, CV_8UC3, full_buffer);
    cv::cvtColor(full_frame, full_frame, cv::COLOR_RGB2BGR);  // OpenCV uses BGR
    
    // FIX: Flip image vertically to correct OpenGL bottom-left origin
    cv::Mat flipped_frame;
    cv::flip(full_frame, flipped_frame, 0);  // 0 = flip around x-axis (vertical flip)
    
    // Resize to target resolution for ViNT
    cv::Mat resized_frame;
    cv::resize(flipped_frame, resized_frame, cv::Size(VINT_IMAGE_WIDTH, VINT_IMAGE_HEIGHT));
    
    // Save frame
    std::stringstream ss;
    ss << g_log_dir << "/frames/frame_" << std::setfill('0') << std::setw(6) << g_frame_counter << ".png";
    cv::imwrite(ss.str(), resized_frame);
    
    // Also save as current_frame.png for preview
    std::string current_frame_path = g_log_dir + "/current_frame.png";
    cv::imwrite(current_frame_path, resized_frame);
    
    delete[] full_buffer;
}

// Log frame data (call from camDraw after grDrawScene)
extern "C" void logVintFrame(tCarElt* car, int scrx, int scry, int scrw, int scrh) {
    if (!g_logging_enabled) return;
    
    uint64_t timestamp = getCurrentTimestamp();
    
    // Save frame
    saveFrame(scrx, scry, scrw, scrh);
    
    // Log pose
    float x = car->pub.DynGCg.pos.x;
    float y = car->pub.DynGCg.pos.y;
    float theta = car->pub.DynGCg.pos.az;
    float speed = car->pub.speed;
    
    g_pose_log << timestamp << ","
               << g_frame_counter << ","
               << x << ","
               << y << ","
               << theta << ","
               << speed << "\n";
    
    // Calculate and log goal
    float goal_x, goal_y;
    calculateGoalPoint(car, goal_x, goal_y);
    
    g_goal_log << timestamp << ","
               << g_frame_counter << ","
               << goal_x << ","
               << goal_y << "\n";
    
    g_frame_counter++;
    
    // Print progress every 100 frames with detailed data
    if (g_frame_counter % 100 == 0) {
        printf("Logged %d frames (%dx%d) | Car pos: (%.2f, %.2f) | Speed: %.2f | Goal: (%.2f, %.2f) | Dir: %s\n", 
               g_frame_counter, VINT_IMAGE_WIDTH, VINT_IMAGE_HEIGHT, x, y, speed, goal_x, goal_y, g_log_dir.c_str());
    }
    
    // Print detailed data every 10 frames for first 50 frames
    if (g_frame_counter <= 50 && g_frame_counter % 10 == 0) {
        printf("DEBUG Frame %d: timestamp=%lu, pos=(%.3f,%.3f), theta=%.3f, speed=%.3f, goal=(%.3f,%.3f)\n",
               g_frame_counter, timestamp, x, y, theta, speed, goal_x, goal_y);
    }
}

// Toggle data logging (call from keyboard handler)
extern "C" void toggleVintLogging() {
    g_logging_enabled = !g_logging_enabled;
    if (g_logging_enabled) {
        printf("ViNT logging enabled\n");
    } else {
        printf("ViNT logging disabled\n");
    }
}

// Check if logging is enabled
extern "C" bool isVintLoggingEnabled() {
    return g_logging_enabled;
}

// Get current log directory
extern "C" const char* getVintLogDirectory() {
    return g_log_dir.c_str();
} 