#include <stdio.h>
#include <string>
#include "vbx_cnn_api.h"
#include "postprocess.h"
#include <math.h>
#include <stdarg.h>

extern "C" int read_JPEG_file(const char *filename, int* width, int* height, unsigned char **image);

void* read_image(const char* filename,int size,int data_type) {
    unsigned char *image;
    int h, w;
    
    read_JPEG_file(filename, &w, &h, &image);
    
    unsigned char *planer_img = (unsigned char *) malloc(w * h * 3);
    
    for (int r = 0; r < h; r++) {
        for (int c = 0; c < w; c++) {
            planer_img[r * w + c] = image[(r * w + c)];
        }
    }

    free(image);
    return planer_img;
}

static inline void print_log(const char *fmt, ...)
{
    va_list args;
    va_start(args, fmt);
    printf("[LOG] ");
    vprintf(fmt, args);
    va_end(args);
}

static inline void print_score(double f)
{
    print_log("[LOG SCORE]=%f\n", f);
}

static inline double apply_sigmoid(double x)
{
    return (1.0 / (1.0 + exp(-x)));
}

static uint32_t fletcher32(const uint16_t *data, size_t len)
{
    uint32_t c0, c1;
    unsigned int i;

    for (c0 = c1 = 0; len >= 360; len -= 360) {
        for (i = 0; i < 360; ++i) {
                c0 = c0 + *data++;
                c1 = c1 + c0;
        }
        c0 = c0 % 65535;
        c1 = c1 % 65535;
    }
    for (i = 0; i < len; ++i) {
            c0 = c0 + *data++;
            c1 = c1 + c0;
    }
    c0 = c0 % 65535;
    c1 = c1 % 65535;
    return (c1 << 16 | c0);
}

int main(int argc, char** argv){

	// On hardware these two variables would be set with real values
	// because this is for the simulator, we use NULL
	void* ctrl_reg_addr = NULL;
	void* firmware_blob = NULL;
	vbx_cnn_t* vbx_cnn = vbx_cnn_init(ctrl_reg_addr,firmware_blob);

	if(argc < 3) {
		printf("Usage %s MODEL_FILE IMAGE.jpg\n", argv[0]);
		return 1;
	}

	FILE* model_file = fopen(argv[1], "r");
	if(model_file == NULL) {
		printf("Unable to open file %s\n", argv[1]);
		return 1;
	}
    
	fseek(model_file, 0, SEEK_END);
	int file_size = ftell(model_file);
	fseek(model_file, 0, SEEK_SET);

	model_t* model = (model_t*) malloc(file_size);
	int size_read = fread(model, 1, file_size, model_file);
	
    if(size_read != file_size){
		fprintf(stderr, "Error reading full model file %s\n", argv[1]);
	}
	
    int model_data_size = model_get_data_bytes(model);
	
    if(model_data_size != file_size){
		fprintf(stderr, "Error model file is not correct size%s\n", argv[1]);
	}
	
    int model_allocate_size = model_get_allocate_bytes(model);
	
    model = (model_t *) realloc(model,model_allocate_size);
	
	uint8_t* input_buffer = NULL;
	void* read_buffer = NULL;
	
    if (std::string(argv[2]) != "TEST_DATA") {
	  int input_datatype = model_get_input_datatype(model, 0);
	  int input_length = model_get_input_length(model, 0);
	  int side = 1;

	  while(side*side*1 < input_length)side++;
	  
      read_buffer = read_image(argv[2],side,input_datatype);
	  input_buffer = (uint8_t*)read_buffer;
	} else {
	  input_buffer = (uint8_t*)model_get_test_input(model, 0);
	}
	int output_length = model_get_output_length(model, 0);
	int output_length1 = 0;
	fix16_t* output_buffer0 = (fix16_t*)malloc(output_length*sizeof(fix16_t));
	fix16_t* output_buffer1 = NULL;

	if (model_get_num_outputs(model) == 2) {
		output_length1 = model_get_output_length(model, 1);
		output_buffer1 = (fix16_t *) malloc(output_length1 * sizeof(fix16_t));
	}

	vbx_cnn_io_ptr_t io_buffers[3] = {(uintptr_t) input_buffer,
	                                  (uintptr_t) output_buffer0,
					                  (uintptr_t) output_buffer1};
	// Buffers are now setup,
	// We can run the model.
	vbx_cnn_model_start(vbx_cnn, model, io_buffers);
	int err=1;
	while (err>0) {
		err = vbx_cnn_model_poll(vbx_cnn);
	}
	if (err<0) {
		printf("Model Run failed with error code: %d\n", err);
	}
	
    // Data should be available int the output buffers now.
	int score = output_buffer0[0];
    double f = (double) score;
    f = score / 65536.0;
    print_log("Before Sigmoid %f\n", f);
    print_score(apply_sigmoid(f));

	int32_t checksum = fletcher32((uint16_t*)io_buffers[1], output_length*2);
	if (io_buffers[2]) {
	  checksum ^= fletcher32((uint16_t*)io_buffers[2], output_length1*2);
	}
	
    printf("CHECKSUM = 0x%08x\n",checksum);
	
    if (read_buffer) {
		free(read_buffer);
	}
	free(model);

	return 0;
}
