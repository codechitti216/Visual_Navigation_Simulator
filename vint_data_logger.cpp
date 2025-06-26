#include <GL/gl.h>
#include <GL/glut.h>
#include <fstream>
#include <sstream>
#include <sys/stat.h>
#include <sys/time.h>
#include <opencv2/opencv.hpp>

// Global logger state
static bool g_logging_enabled = false;
static std::string g_log_dir;
static std::ofstream g_pose_log;
static std::ofstream g_goal_log;
static uint32_t g_frame_counter = 0;
static uint64_t g_session_start_time = 0;

// Create directory if it doesn't exist
bool createDirectory(const std::string& path) {
    struct stat st = {0};
    if (stat(path.c_str(), &st) == -1) {
        return mkdir(path.c_str(), 0700) == 0;
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
    ss << "/data/vint_training_logs/" << track_name << "/" << g_session_start_time;
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
        metadata_log << "  \"frame_resolution\": [224, 224],\n";
        metadata_log << "  \"lookahead_distance\": 10.0,\n";
        metadata_log << "  \"start_time\": " << g_session_start_time << "\n";
        metadata_log << "}\n";
        metadata_log.close();
    }
    
    g_logging_enabled = true;
    g_frame_counter = 0;
    
    printf("ViNT data logger initialized: %s\n", g_log_dir.c_str());
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
    
    // Convert to OpenCV Mat
    cv::Mat full_frame(scrh, scrw, CV_8UC3, full_buffer);
    cv::cvtColor(full_frame, full_frame, cv::COLOR_RGB2BGR);  // OpenCV uses BGR
    
    // Resize to 224x224
    cv::Mat resized_frame;
    cv::resize(full_frame, resized_frame, cv::Size(224, 224));
    
    // Save frame
    std::stringstream ss;
    ss << g_log_dir << "/frames/frame_" << std::setfill('0') << std::setw(6) << g_frame_counter << ".png";
    cv::imwrite(ss.str(), resized_frame);
    
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
    
    // Print progress every 100 frames
    if (g_frame_counter % 100 == 0) {
        printf("Logged %d frames\n", g_frame_counter);
    }
}

// Toggle logging on/off
extern "C" void toggleVintLogging() {
    g_logging_enabled = !g_logging_enabled;
    printf("ViNT logging %s\n", g_logging_enabled ? "enabled" : "disabled");
}

// Get current logging status
extern "C" bool isVintLoggingEnabled() {
    return g_logging_enabled;
}

// Get current log directory
extern "C" const char* getVintLogDirectory() {
    return g_log_dir.c_str();
} 