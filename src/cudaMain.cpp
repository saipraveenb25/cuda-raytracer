#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include <string>

//#include "refRenderer.h"
#include "cudaRenderer.h"
#include "platformgl.h"


void startRendererWithDisplay(cutracer::CudaRenderer* renderer);
//void startBenchmark(CircleRenderer* renderer, int startFrame, int totalFrames, const std::string& frameFilename);
//void CheckBenchmark(CircleRenderer* ref_renderer, CircleRenderer* cuda_renderer,
//                        int benchmarkFrameStart, int totalFrames, const std::string& frameFilename);


void usage(const char* progname) {
    printf("Usage: %s [options] scenefile\n", progname);
    printf("scenefile should be an XML file that describes the scene\n");
    printf("Program Options:\n");
    printf("  -b  --bench <START:END>    Benchmark mode, do not create display. Time frames [START,END)\n");
    printf("  -c  --check                Check correctness of output\n");
    printf("  -f  --file  <FILENAME>     Dump frames in benchmark mode (FILENAME_xxxx.ppm)\n");
    printf("  -r  --renderer <ref/cuda>  Select renderer: ref or cuda\n");
    printf("  -s  --size  <INT>          Make rendered image <INT>x<INT> pixels\n");
    printf("  -?  --help                 This message\n");
}


int main(int argc, char** argv)
{

    int benchmarkFrameStart = -1;
    int benchmarkFrameEnd = -1;
    int imageSize = IMAGE_SIZE;
    
    if(argc <= 1) {
        usage(argv[0]);
    }

    std::string sceneNameStr(argv[1]);
    std::string frameFilename;
    // parse commandline options ////////////////////////////////////////////

    /*while ((opt = getopt_long(argc, argv, "b:f:r:s:c?", long_options, NULL)) != EOF) {

      switch (opt) {
      case 'b':
      if (sscanf(optarg, "%d:%d", &benchmarkFrameStart, &benchmarkFrameEnd) != 2) {
      fprintf(stderr, "Invalid argument to -b option\n");
      usage(argv[0]);
      exit(1);
      }
      break;
      case 'c':
      checkCorrectness = true;
      break;
      case 'f':
      frameFilename = optarg;
      break;
      case 'r':
      if (std::string(optarg).compare("cuda") == 0) {
      useRefRenderer = false;
      }
      break;
      case 's':
      imageSize = atoi(optarg);
      break;
      case '?':
      default:
      usage(argv[0]);
      return 1;
      }
      }*/
    // end parsing of commandline options ////////////////////////////////////// 


    if (optind + 1 > argc) {
        fprintf(stderr, "Error: missing scene name\n");
        usage(argv[0]);
        return 1;
    }

    printf("Rendering to %dx%d image\n", imageSize, imageSize);

    cutracer::CudaRenderer* renderer;

    renderer = new cutracer::CudaRenderer();

    renderer->allocOutputImage(imageSize, imageSize);
    renderer->loadScene(sceneNameStr);
    renderer->setup();

    //if (benchmarkFrameStart >= 0){
    //    startBenchmark(renderer, benchmarkFrameStart, benchmarkFrameEnd - benchmarkFrameStart, frameFilename);
        //printf("Done with benchmark.");
    //} else {
    glutInit(&argc, argv);
    startRendererWithDisplay(renderer);
    //}


    return 0;
}
