#ifndef _VINT_DATA_LOGGER_H_
#define _VINT_DATA_LOGGER_H_

#include <car.h>

#ifdef __cplusplus
extern "C" {
#endif

// Initialize data logger with track name
int initVintDataLogger(const char* track_name);

// Cleanup data logger
void cleanupVintDataLogger();

// Log frame data (call from camDraw after grDrawScene)
void logVintFrame(tCarElt* car, int scrx, int scry, int scrw, int scrh);

// Toggle logging on/off
void toggleVintLogging();

// Get current logging status
bool isVintLoggingEnabled();

// Get current log directory
const char* getVintLogDirectory();

#ifdef __cplusplus
}
#endif

#endif /* _VINT_DATA_LOGGER_H_ */ 