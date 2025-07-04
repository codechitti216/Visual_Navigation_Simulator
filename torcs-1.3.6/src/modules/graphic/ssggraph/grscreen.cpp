/***************************************************************************

    file        : grscreen.cpp
    created     : Thu May 15 22:11:03 CEST 2003
    copyright   : (C) 2003 by Eric Espié
    email       : eric.espie@torcs.org
    version     : $Id: grscreen.cpp,v 1.22.2.5 2012/06/09 11:44:46 berniw Exp $

 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

/** @file

    @author	<a href=mailto:eric.espie@torcs.org>Eric Espie</a>
    @version	$Id: grscreen.cpp,v 1.22.2.5 2012/06/09 11:44:46 berniw Exp $
*/

#include <plib/ssg.h>

#include <tgfclient.h>
#include <portability.h>
#include "grutil.h"
#include "grmain.h"
#include "grscene.h"
#include "grshadow.h"
#include "grskidmarks.h"
#include "grcar.h"
#include "grboard.h"
#include "grcarlight.h"

#include "grscreen.h"
#include "vint_data_logger.h"

cGrScreen::cGrScreen(int myid)
{
	id = myid;
	curCar = NULL;
	curCam = NULL;
	mirrorCam = NULL;
	dispCam = NULL;
	boardCam = NULL;
	bgCam = NULL;
	board = NULL;
	curCamHead = 0;
	drawCurrent = 0;
	active = 0;
	selectNextFlag = 0;
	selectPrevFlag = 0;
	mirrorFlag = 1;
	memset(cams, 0, sizeof(cams));
	viewRatio = 1.33;
	cars = 0;
}

cGrScreen::~cGrScreen()
{
	unsigned int i;
	class cGrCamera *cam;

	for (i = 0; i < sizeof(cams)/sizeof(cams[0]); i++) {
		while ((cam =  GF_TAILQ_FIRST(&cams[i])) != 0) {
			cam->remove(&cams[i]);
			delete cam;
		}
	}

	delete boardCam;
	delete mirrorCam;
	delete bgCam;

	if (board != NULL) {
		board->shutdown ();
	}

	FREEZ(cars);

	if (board != NULL) {
		delete board;
		board = NULL;
	}
}

int cGrScreen::isInScreen(int x, int y)
{
	if (!active) {
		return 0;
	}

	if ((x >= scrx) &&
		(y >= scry) &&
		(x < (scrx + scrw)) &&
		(y < (scry + scrh)))
	{
		return 1;
	}

	return 0;
}

void cGrScreen::setCurrentCar(tCarElt *newCurCar)
{
	curCar = newCurCar;
}

/* Set Screen size & position */
void cGrScreen::activate(int x, int y, int w, int h)
{
	viewRatio = (float)w / (float)h;

	scrx = x;
	scry = y;
	scrw = w;
	scrh = h;

	if (mirrorCam) {
		mirrorCam->setViewport (scrx, scry, scrw, scrh);
		mirrorCam->setPos (scrx + scrw / 4, scry +  5 * scrh / 6 - scrh / 10, scrw / 2, scrh / 6);
	}
	if (curCam) {
		curCam->limitFov ();
		curCam->setZoom (GR_ZOOM_DFLT);
	}
	active = 1;
}

void cGrScreen::desactivate(void)
{
	active = 0;
}

/* Set camera zoom value */
void cGrScreen::setZoom(long zoom)
{
	curCam->setZoom(zoom);
}

void cGrScreen::selectNextCar(void)
{
	selectNextFlag = 1;
}

void cGrScreen::selectPrevCar(void)
{
	selectPrevFlag = 1;
}

void cGrScreen::selectBoard(long brd)
{
	board->selectBoard(brd);
}

void cGrScreen::selectTrackMap()
{
	const int BUFSIZE=1024;
	char path[BUFSIZE];
	char path2[BUFSIZE];
	int viewmode;

	board->getTrackMap()->selectTrackMap();
	viewmode = board->getTrackMap()->getViewMode();

	snprintf(path, BUFSIZE, "%s/%d", GR_SCT_DISPMODE, id);
	GfParmSetNum(grHandle, path, GR_ATT_MAP, NULL, (tdble)viewmode);
	/* save also as user's preference if human */
	if (curCar->_driverType == RM_DRV_HUMAN) {
		snprintf(path2, BUFSIZE, "%s/%s", GR_SCT_DISPMODE, curCar->_name);
		GfParmSetNum(grHandle, path2, GR_ATT_MAP, NULL, (tdble)viewmode);
	}
	GfParmWriteFile(NULL, grHandle, "Graph");
}

void cGrScreen::switchMirror(void)
{
	const int BUFSIZE=1024;
	char path[BUFSIZE];
	char path2[BUFSIZE];

	mirrorFlag = 1 - mirrorFlag;
	snprintf(path, BUFSIZE, "%s/%d", GR_SCT_DISPMODE, id);
	GfParmSetNum(grHandle, path, GR_ATT_MIRROR, NULL, (tdble)mirrorFlag);
	/* save also as user's preference if human */
	if (curCar->_driverType == RM_DRV_HUMAN) {
		snprintf(path2, BUFSIZE, "%s/%s", GR_SCT_DISPMODE, curCar->_name);
		GfParmSetNum(grHandle, path, GR_ATT_MIRROR, NULL, (tdble)mirrorFlag);
	}
	GfParmWriteFile(NULL, grHandle, "Graph");
}


/* Select the camera by number */
void cGrScreen::selectCamera(long cam)
{
	const int BUFSIZE=1024;
	char buf[BUFSIZE];
	char path[BUFSIZE];
	char path2[BUFSIZE];

	// NEW: Print when camera changes
	printf("Camera Change: curCamHead changing from %d to %d\n", curCamHead, cam);

	if (cam == curCamHead) {
		/* Same camera list, choose the next one */
		curCam = curCam->next();
		if (curCam == (cGrPerspCamera*)GF_TAILQ_END(&cams[cam])) {
			curCam = (cGrPerspCamera*)GF_TAILQ_FIRST(&cams[cam]);
		}
	} else {
		/* Change of camera list, take the first one */
		curCamHead = cam;
		curCam = (cGrPerspCamera*)GF_TAILQ_FIRST(&cams[cam]);
	}

	if (curCam == NULL) {
		/* back to default camera */
		curCamHead = 0;
		curCam = (cGrPerspCamera*)GF_TAILQ_FIRST(&cams[curCamHead]);
	}

	snprintf(path, BUFSIZE, "%s/%d", GR_SCT_DISPMODE, id);
	GfParmSetStr(grHandle, path, GR_ATT_CUR_DRV, curCar->_name);
	GfParmSetNum(grHandle, path, GR_ATT_CAM, (char*)NULL, (tdble)curCam->getId());
	GfParmSetNum(grHandle, path, GR_ATT_CAM_HEAD, (char*)NULL, (tdble)curCamHead);

	/* save also as user's preference if human */
	if (curCar->_driverType == RM_DRV_HUMAN) {
		snprintf(path2, BUFSIZE, "%s/%s", GR_SCT_DISPMODE, curCar->_name);
		GfParmSetNum(grHandle, path2, GR_ATT_CAM, (char*)NULL, (tdble)curCam->getId());
		GfParmSetNum(grHandle, path2, GR_ATT_CAM_HEAD, (char*)NULL, (tdble)curCamHead);
	}

	snprintf(buf, BUFSIZE, "%s-%d-%d", GR_ATT_FOVY, curCamHead, curCam->getId());
	curCam->loadDefaults(buf);
	drawCurrent = curCam->getDrawCurrent();
	curCam->limitFov ();
	GfParmWriteFile(NULL, grHandle, "Graph");
}

static class cGrPerspCamera *ThedispCam;	/* the display camera */

static int
comparCars(const void *car1, const void *car2)
{
	float d1 = ThedispCam->getDist2(*(tCarElt**)car1);
	float d2 = ThedispCam->getDist2(*(tCarElt**)car2);

	if (d1 > d2) {
		return -1;
	} else {
		return 1;
	}
}

// Yurong
extern int drawIndicator;
// Yurong

void cGrScreen::camDraw(tSituation *s)
{
	int i;

	glDisable(GL_COLOR_MATERIAL);

	START_PROFILE("dispCam->update*");
	dispCam->update(curCar, s);
	STOP_PROFILE("dispCam->update*");

	if (dispCam->getDrawBackground()) {
		glDisable(GL_LIGHTING);
		glDisable(GL_DEPTH_TEST);
		grDrawBackground(dispCam, bgCam);
		glClear(GL_DEPTH_BUFFER_BIT);
	}
	glEnable(GL_DEPTH_TEST);

	START_PROFILE("dispCam->action*");
	dispCam->action();
	STOP_PROFILE("dispCam->action*");

	START_PROFILE("grDrawCar*");
	glFogf(GL_FOG_START, dispCam->getFogStart());
	glFogf(GL_FOG_END, dispCam->getFogEnd());
	glEnable(GL_FOG);

	/*sort the cars by distance for transparent windows */
	ThedispCam = dispCam;
	qsort(cars, s->_ncars, sizeof(tCarElt*), comparCars);

	for (i = 0; i < s->_ncars; i++) {
		grDrawCar(cars[i], curCar, dispCam->getDrawCurrent(), dispCam->getDrawDriver(), s->currentTime, dispCam);
	}

	STOP_PROFILE("grDrawCar*");

	START_PROFILE("grDrawScene*");
	glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	grDrawScene();
	STOP_PROFILE("grDrawScene*");

	// ViNT Data Logger: Capture frame after scene is rendered
	if (curCar && curCar->_driverType == RM_DRV_HUMAN) {
		// Auto-switch to ego-centric cockpit camera when logging
		extern bool isVintLoggingEnabled();
		static bool logging_camera_forced = false;
		static int original_cam_head = -1;
		static int original_cam_id = -1;
		
		bool logging_enabled = isVintLoggingEnabled();
		
		// Print camera options when logging is enabled
		if (logging_enabled) {
			static int camera_list_print_counter = 0;
			
			// Print camera list every 60 frames (about once per second)
			if (camera_list_print_counter % 60 == 0) {
				// COMMENTED OUT: printf("ViNT: Current camera - Head=%d, ID=%d\n", curCamHead, curCam->getId());
				// COMMENTED OUT: printf("ViNT: === AVAILABLE CAMERA VIEWS ===\n");
				
				// COMMENTED OUT: Print all available cameras in all sections
				// COMMENTED OUT: for (int head = 0; head < 3; head++) {
				// COMMENTED OUT: 	printf("ViNT: --- Camera Head %d (F%d key) ---\n", head, head+1);
				// COMMENTED OUT: 	class cGrCamera *cam = GF_TAILQ_FIRST(&cams[head]);
				// COMMENTED OUT: 	int cam_count = 0;
				// COMMENTED OUT: 	while (cam) {
				// COMMENTED OUT: 		printf("ViNT:   ID=%d (Camera %d in this head)\n", cam->getId(), cam_count);
				// COMMENTED OUT: 		cam = cam->next();
				// COMMENTED OUT: 		cam_count++;
				// COMMENTED OUT: 	}
				// COMMENTED OUT: 	if (cam_count == 0) {
				// COMMENTED OUT: 		printf("ViNT:   (No cameras in this head)\n");
				// COMMENTED OUT: 	}
				// COMMENTED OUT: }
				// COMMENTED OUT: printf("ViNT: ===========================\n");
				// COMMENTED OUT: printf("ViNT: Please tell me which Head and ID you want!\n");
				// COMMENTED OUT: printf("ViNT: (e.g., 'Head=1, ID=2' or 'F2, Camera 2')\n");
				
				// NEW: Simple print statement for curCamHead variable
				printf("Camera Debug: curCamHead = %d, Camera ID = %d\n", curCamHead, curCam->getId());
				fflush(stdout);
			}
			camera_list_print_counter++;
			
			// Save current camera settings if not done already
			if (!logging_camera_forced) {
				original_cam_head = curCamHead;
				original_cam_id = curCam->getId();
				logging_camera_forced = true;
			}
		}
		// Restore original camera when logging stops
		else if (!logging_enabled && logging_camera_forced) {
			// Restore original camera
			if (original_cam_head >= 0 && original_cam_id >= 0) {
				class cGrCamera *cam = GF_TAILQ_FIRST(&cams[original_cam_head]);
				while (cam) {
					if (cam->getId() == original_cam_id) {
						curCamHead = original_cam_head;
						curCam = (cGrPerspCamera*)cam;
						curCam->onSelect(curCar, s);
						break;
					}
					cam = cam->next();
				}
			}
			logging_camera_forced = false;
			// COMMENTED OUT: printf("ViNT: Restored original camera view\n");
		}
		
		logVintFrame(curCar, scrx, scry, scrw, scrh);
	}
}


/* Update screen display */
void cGrScreen::update(tSituation *s, float Fps)
{
	int i;
	//ssgLight *light;
	int carChanged;
	const int BUFSIZE=1024;
	char buf[BUFSIZE];

	if (!active) {
		return;
	}

	carChanged = 0;
	if (selectNextFlag) {
		for (i = 0; i < (s->_ncars - 1); i++) {
			if (curCar == s->cars[i]) {
				curCar = s->cars[i + 1];
				curCar->priv.collision = 0;
				carChanged = 1;
				break;
			}
		}
		selectNextFlag = 0;
	}

	if (selectPrevFlag) {
		for (i = 1; i < s->_ncars; i++) {
			if (curCar == s->cars[i]) {
				curCar = s->cars[i - 1];
				curCar->priv.collision = 0;
				carChanged = 1;
				break;
			}
		}
		selectPrevFlag = 0;
	}
	if (carChanged) {
		snprintf(buf, BUFSIZE, "%s/%d", GR_SCT_DISPMODE, id);
		GfParmSetStr(grHandle, buf, GR_ATT_CUR_DRV, curCar->_name);
		loadParams (s);
		GfParmWriteFile(NULL, grHandle, "Graph");
		curCam->onSelect(curCar, s);
	}

	//light = ssgGetLight (0);

	/* MIRROR */
	if (mirrorFlag && curCam->isMirrorAllowed ()) {
		mirrorCam->activateViewport ();
		dispCam = mirrorCam;
		glClear (GL_DEPTH_BUFFER_BIT);
		camDraw (s);
		mirrorCam->store ();
	}

	glViewport(scrx, scry, scrw, scrh);
	dispCam = curCam;
	camDraw(s);

	glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);
	glDisable(GL_CULL_FACE);
	glDisable(GL_DEPTH_TEST);
	glDisable(GL_LIGHTING);
	glDisable(GL_COLOR_MATERIAL);
	glDisable(GL_ALPHA_TEST);
	glDisable(GL_FOG);
	glEnable(GL_TEXTURE_2D);

	/* MIRROR */
	if (mirrorFlag && curCam->isMirrorAllowed ()) {
		mirrorCam->display ();
		glViewport (scrx, scry, scrw, scrh);
	}

	START_PROFILE("boardCam*");
	boardCam->action();
	STOP_PROFILE("boardCam*");

	START_PROFILE("grDisp**");
	glDisable(GL_TEXTURE_2D);

	TRACE_GL("cGrScreen::update glDisable(GL_DEPTH_TEST)");
	board->refreshBoard(s, Fps, 0, curCar);
	TRACE_GL("cGrScreen::update display boards");

	STOP_PROFILE("grDisp**");
}

void cGrScreen::loadParams(tSituation *s)
{
	int camNum;
	int i;
	class cGrCamera *cam;
	const char *carName;
	const int BUFSIZE=1024;
	char buf[BUFSIZE];
	char path[BUFSIZE];
	char path2[BUFSIZE];

	snprintf(path, BUFSIZE, "%s/%d", GR_SCT_DISPMODE, id);

	if (!curCar) {
		carName = GfParmGetStr(grHandle, path, GR_ATT_CUR_DRV, "");
		for (i = 0; i < s->_ncars; i++) {
			if (!strcmp(s->cars[i]->_name, carName)) {
				break;
			}
		}
		if (i < s->_ncars) {
			curCar = s->cars[i];
		} else if (id < s->_ncars) {
			curCar = s->cars[id];
		} else {
			curCar = s->cars[0];
		}
	}
	snprintf(path2, BUFSIZE, "%s/%s", GR_SCT_DISPMODE, curCar->_name);

	// DEFAULT: Set to curCamHead = 0, Camera ID = 4 (road view without car)
	curCamHead	= (int)GfParmGetNum(grHandle, path, GR_ATT_CAM_HEAD, NULL, 0);  // Changed from 9 to 0
	camNum	= (int)GfParmGetNum(grHandle, path, GR_ATT_CAM, NULL, 4);  // Changed from 0 to 4
	mirrorFlag	= (int)GfParmGetNum(grHandle, path, GR_ATT_MIRROR, NULL, (tdble)mirrorFlag);
	curCamHead	= (int)GfParmGetNum(grHandle, path2, GR_ATT_CAM_HEAD, NULL, (tdble)curCamHead);
	camNum	= (int)GfParmGetNum(grHandle, path2, GR_ATT_CAM, NULL, (tdble)camNum);
	mirrorFlag	= (int)GfParmGetNum(grHandle, path2, GR_ATT_MIRROR, NULL, (tdble)mirrorFlag);

	cam = GF_TAILQ_FIRST(&cams[curCamHead]);
	curCam = NULL;
	while (cam) {
		if (cam->getId() == camNum) {
			curCam = (cGrPerspCamera*)cam;
			break;
		}
		cam = cam->next();
	}

	if (curCam == NULL) {
		// DEFAULT FALLBACK: Set to curCamHead = 0, Camera ID = 4 (road view without car)
		curCamHead = 0;
		cam = GF_TAILQ_FIRST(&cams[curCamHead]);
		while (cam) {
			if (cam->getId() == 4) {
				curCam = (cGrPerspCamera*)cam;
				break;
			}
			cam = cam->next();
		}
		// If ID 4 doesn't exist, fallback to first camera in head 0
		if (curCam == NULL) {
		curCam = (cGrPerspCamera*)GF_TAILQ_FIRST(&cams[curCamHead]);
		}
		GfParmSetNum(grHandle, path, GR_ATT_CAM, NULL, (tdble)curCam->getId());
		GfParmSetNum(grHandle, path, GR_ATT_CAM_HEAD, NULL, (tdble)curCamHead);
	}

	snprintf(buf, BUFSIZE, "%s-%d-%d", GR_ATT_FOVY, curCamHead, curCam->getId());
	curCam->loadDefaults(buf);
	drawCurrent = curCam->getDrawCurrent();
	board->loadDefaults(curCar);
}


/* Create cameras */
void cGrScreen::initCams(tSituation *s)
{
	tdble fovFactor;
	int i;

	fovFactor = GfParmGetNum(grHandle, GR_SCT_GRAPHIC, GR_ATT_FOVFACT, (char*)NULL, 1.0);
	fovFactor *= GfParmGetNum(grTrackHandle, TRK_SECT_GRAPH, TRK_ATT_FOVFACT, (char*)NULL, 1.0);

	if (boardCam == NULL) {
		boardCam = new cGrOrthoCamera(this,0, grWinw*600/grWinh, 0, 600);
	}

	if (bgCam == NULL) {
		bgCam = new cGrBackgroundCam(this);
	}

	if (mirrorCam == NULL) {
		mirrorCam = new cGrCarCamMirror(
			this,
			-1,
			0,					// drawCurr
			1,					// drawBG
			90.0,				// fovy
			0.0,				// fovymin
			360.0,				// fovymax
			0.3,				// near
			300.0 * fovFactor,	// far
			200.0 * fovFactor,	// fog1
			300.0 * fovFactor	// fog
		);
	}

	// Scene Cameras
	unsigned int j;
	class cGrCamera *cam;
	for (j = 0; j < sizeof(cams)/sizeof(cams[0]); j++) {
		while ((cam = GF_TAILQ_FIRST(&cams[j])) != 0) {
			cam->remove(&cams[j]);
			delete cam;
		}
	}
	memset(cams, 0, sizeof(cams));

	grCamCreateSceneCameraList(this, cams, fovFactor);

	cars = (tCarElt**)calloc(s->_ncars, sizeof (tCarElt*));
	for (i = 0; i < s->_ncars; i++) {
		cars[i] = s->cars[i];
	}

	loadParams(s);
}

void cGrScreen::initBoard(void)
{
	if (board == NULL) {
		board = new cGrBoard (id);
	}
	board->initBoard ();
}
