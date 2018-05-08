#include <algorithm>

#include "cudaRenderer.h"
#include "cycleTimer.h"
#include "cuda_image.h"
#include "platformgl.h"


void renderPicture();


static struct {
    int width;
    int height;
    bool updateSim;
    bool printStats;
    bool pauseSim;
    double lastFrameTime;

    cutracer::CudaRenderer* renderer;

} gDisplay;

// handleReshape --
//
// Event handler, fired when the window is resized
void
handleReshape(int w, int h) {
    gDisplay.width = w;
    gDisplay.height = h;
    glViewport(0, 0, gDisplay.width, gDisplay.height);
    glutPostRedisplay();
}

/*void handleKeyPress(int key, int x, int y) 
{
    switch (key) 
    {    
        case 27 :      break;
        case 100 : printf("GLUT_KEY_LEFT %d\n",key);   break;
        case 102: printf("GLUT_KEY_RIGHT %d\n",key);   break;
        case 101   : printf("GLUT_KEY_UP %d\n",key);   break;
        case 103 : printf("GLUT_KEY_DOWN %d\n",key);   break;
    }

}*/

void
handleDisplay() {

    // simulation and rendering work is done in the renderPicture
    // function below

    renderPicture();

    // the subsequent code uses OpenGL to present the state of the
    // rendered image on the screen.

    const Image* img = gDisplay.renderer->getImage();

    int width = std::min(img->width, gDisplay.width);
    int height = std::min(img->height, gDisplay.height);

    glDisable(GL_DEPTH_TEST);
    glClearColor(0.f, 0.f, 0.f, 1.f);
    glClear(GL_COLOR_BUFFER_BIT);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.f, gDisplay.width, 0.f, gDisplay.height, -1.f, 1.f);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    // copy image data from the renderer to the OpenGL
    // frame-buffer.  This is inefficient solution is the processing
    // to generate the image is done in CUDA.  An improved solution
    // would render to a CUDA surface object (stored in GPU memory),
    // and then bind this surface as a texture enabling it's use in
    // normal openGL rendering
    glRasterPos2i(0, 0);
    glDrawPixels(width, height, GL_RGBA, GL_FLOAT, img->data);

    double currentTime = CycleTimer::currentSeconds();

    if (gDisplay.printStats)
        printf("%.2f ms\n", 1000.f * (currentTime - gDisplay.lastFrameTime));

    gDisplay.lastFrameTime = currentTime;

    glutSwapBuffers();
    glutPostRedisplay();
}


// handleKeyPress --
//
// Keyboard event handler
void
handleKeyPress(unsigned char key, int x, int y) {

    switch (key) {
        case 'q':
        case 'Q':
            exit(1);
            break;
        case '=':
        case '+':
            gDisplay.updateSim = true;
            break;
        case 'p':
        case 'P':
            gDisplay.pauseSim = !gDisplay.pauseSim;
            if (!gDisplay.pauseSim)
                gDisplay.updateSim = true;
            break;

        case 'w':
        case 'W':
            gDisplay.renderer->c_origin += Vector3D(0, 0, -0.01);
            gDisplay.renderer->setViewpoint(gDisplay.renderer->c_origin, gDisplay.renderer->c_lookAt);
            break;
        case 's':
        case 'S':
            gDisplay.renderer->c_origin += Vector3D(0, 0, 0.01);
            gDisplay.renderer->setViewpoint(gDisplay.renderer->c_origin, gDisplay.renderer->c_lookAt);
            break;
        case 'a':
        case 'A':
            gDisplay.renderer->c_origin += Vector3D(-0.01, 0, 0);
            gDisplay.renderer->setViewpoint(gDisplay.renderer->c_origin, gDisplay.renderer->c_lookAt);
            break;
        case 'd':
        case 'D':
            gDisplay.renderer->c_origin += Vector3D(0.01, 0, 0);
            gDisplay.renderer->setViewpoint(gDisplay.renderer->c_origin, gDisplay.renderer->c_lookAt);
            break;
    }
}

// renderPicture --
//
// At the reall work is done here, not in the display handler
void
renderPicture() {

    //sleep(3000);
    //int a;
    //std::cin >> a;
    double startTime = CycleTimer::currentSeconds();

    // clear screen
    //printf("Clearing\n");fflush(stdout);
    gDisplay.renderer->clearImage();
    //printf("Clearing done\n");fflush(stdout);

    double endClearTime = CycleTimer::currentSeconds();

    // update particle positions and state
    //if (gDisplay.updateSim) {
    //printf("Advancing\n");fflush(stdout);
    //    gDisplay.renderer->advanceAnimation();
    //printf("Advancing done\n");fflush(stdout);
    //}

    if (gDisplay.pauseSim)
        gDisplay.updateSim = false;

    double endSimTime = CycleTimer::currentSeconds();

    // render the particles< into the image
    //printf("Rendering\n");fflush(stdout);
    gDisplay.renderer->render();
    //printf("Rendering done\n");fflush(stdout);

    double endRenderTime = CycleTimer::currentSeconds();

    if (gDisplay.printStats) {
        printf("Clear:    %.3f ms\n", 1000.f * (endClearTime - startTime));
        printf("Advance:  %.3f ms\n", 1000.f * (endSimTime - endClearTime));
        printf("Render:   %.3f ms\n", 1000.f * (endRenderTime - endSimTime));
    }
    
    #ifndef RENDER_ACCUMULATE
    int a;
    std::cin >> a;
    #endif
    //int a;
    //std::cin >> a;
}

void
startRendererWithDisplay(cutracer::CudaRenderer* renderer) {

    // setup the display

    const Image* img = renderer->getImage();

    gDisplay.renderer = renderer;
    gDisplay.updateSim = true;
    gDisplay.pauseSim = false;
    gDisplay.printStats = true;
    gDisplay.lastFrameTime = CycleTimer::currentSeconds();
    gDisplay.width = img->width;
    gDisplay.height = img->height;

    // configure GLUT

    glutInitWindowSize(gDisplay.width, gDisplay.height);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutCreateWindow("CMU 15-618 FINAL PROJECT");
    glutDisplayFunc(handleDisplay);
    glutKeyboardFunc(handleKeyPress);
    glutMainLoop();
}
