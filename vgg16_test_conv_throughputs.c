#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include "conv.h"
#include "basic.h"
#include "pool.h"
#include "FPGA_DDR.h"

#include "vgg16_mp_Tin32Tout32_cfg.h"
#include "vgg16_mp_Tin32Tout32_scale.h"


int main()
{
    unsigned long int time;

    printf("Hello World Vgg16\n");
    FPGA_Init();

	struct Mapped_Feature* image = Malloc_Feature(conv_Hin[0], conv_Win[0], conv_CHin[0], conv_in_scale[0], conv_in_scale[0], conv_in_bit[0]);
	struct Mapped_Feature* conv0_relu_out = Malloc_Feature(conv_Hout[0], conv_Wout[0], conv_CHout[0], conv_out_scale[0], conv_out_scale[0], conv_out_bit[0]);
	struct Mapped_Feature* conv1_relu_out = Malloc_Feature(conv_Hout[1], conv_Wout[1], conv_CHout[1], conv_out_scale[1], conv_out_scale[1], conv_out_bit[1]);
	struct Mapped_Feature* pool0_out = Malloc_Feature(pool_Hout[0], pool_Wout[0], pool_CHout[0], pool_out_scale[0], pool_out_scale[0], pool_out_bit[0]);
	struct Mapped_Feature* conv2_relu_out = Malloc_Feature(conv_Hout[2], conv_Wout[2], conv_CHout[2], conv_out_scale[2], conv_out_scale[2], conv_out_bit[2]);
	struct Mapped_Feature* conv3_relu_out = Malloc_Feature(conv_Hout[3], conv_Wout[3], conv_CHout[3], conv_out_scale[3], conv_out_scale[3], conv_out_bit[3]);
	struct Mapped_Feature* pool1_out = Malloc_Feature(pool_Hout[1], pool_Wout[1], pool_CHout[1], pool_out_scale[1], pool_out_scale[1], pool_out_bit[1]);
	struct Mapped_Feature* conv4_relu_out = Malloc_Feature(conv_Hout[4], conv_Wout[4], conv_CHout[4], conv_out_scale[4], conv_out_scale[4], conv_out_bit[4]);
	struct Mapped_Feature* conv5_relu_out = Malloc_Feature(conv_Hout[5], conv_Wout[5], conv_CHout[5], conv_out_scale[5], conv_out_scale[5], conv_out_bit[5]);
	struct Mapped_Feature* conv6_relu_out = Malloc_Feature(conv_Hout[6], conv_Wout[6], conv_CHout[6], conv_out_scale[6], conv_out_scale[6], conv_out_bit[6]);
	struct Mapped_Feature* pool2_out = Malloc_Feature(pool_Hout[2], pool_Wout[2], pool_CHout[2], pool_out_scale[2], pool_out_scale[2], pool_out_bit[2]);
	struct Mapped_Feature* conv7_relu_out = Malloc_Feature(conv_Hout[7], conv_Wout[7], conv_CHout[7], conv_out_scale[7], conv_out_scale[7], conv_out_bit[7]);
	struct Mapped_Feature* conv8_relu_out = Malloc_Feature(conv_Hout[8], conv_Wout[8], conv_CHout[8], conv_out_scale[8], conv_out_scale[8], conv_out_bit[8]);
	struct Mapped_Feature* conv9_relu_out = Malloc_Feature(conv_Hout[9], conv_Wout[9], conv_CHout[9], conv_out_scale[9], conv_out_scale[9], conv_out_bit[9]);
	struct Mapped_Feature* pool3_out = Malloc_Feature(pool_Hout[3], pool_Wout[3], pool_CHout[3], pool_out_scale[3], pool_out_scale[3], pool_out_bit[3]);
	struct Mapped_Feature* conv10_relu_out = Malloc_Feature(conv_Hout[10], conv_Wout[10], conv_CHout[10], conv_out_scale[10], conv_out_scale[10], conv_out_bit[10]);
	struct Mapped_Feature* conv11_relu_out = Malloc_Feature(conv_Hout[11], conv_Wout[11], conv_CHout[11], conv_out_scale[11], conv_out_scale[11], conv_out_bit[11]);
	struct Mapped_Feature* conv12_relu_out = Malloc_Feature(conv_Hout[12], conv_Wout[12], conv_CHout[12], conv_out_scale[12], conv_out_scale[12], conv_out_bit[12]);
	struct Mapped_Feature* pool4_out = Malloc_Feature(pool_Hout[4], pool_Wout[4], pool_CHout[4], pool_out_scale[4], pool_out_scale[4], pool_out_bit[4]);
	struct Mapped_Feature* conv13_relu_out = Malloc_Feature(conv_Hout[13], conv_Wout[13], conv_CHout[13], conv_out_scale[13], conv_out_scale[13], conv_out_bit[13]);
	struct Mapped_Feature* conv14_relu_out = Malloc_Feature(conv_Hout[14], conv_Wout[14], conv_CHout[14], conv_out_scale[14], conv_out_scale[14], conv_out_bit[14]);
	struct Mapped_Feature* conv15_relu_out = Malloc_Feature(conv_Hout[15], conv_Wout[15], conv_CHout[15], conv_out_scale[15], conv_out_scale[15], conv_out_bit[15]);
	printf("Malloc Memory Done\n");

	struct Mapped_Weight* conv0_W = Malloc_Weight(conv_Ky[0], conv_Kx[0], conv_CHin[0], conv_CHout[0], conv_wt_scale[0],conv_wt_bit[0]);
	Load_Weight_From_bin_File(conv0_W, "Vgg16_weight/conv0_8b_quant_s8.bin");
	
	struct Mapped_Weight* conv1_W = Malloc_Weight(conv_Ky[1], conv_Kx[1], conv_CHin[1], conv_CHout[1], conv_wt_scale[1],conv_wt_bit[1]);
	Load_Weight_From_bin_File(conv1_W, "Vgg16_weight/conv1_8b_quant_s10.bin");
	
	struct Mapped_Weight* conv2_W = Malloc_Weight(conv_Ky[2], conv_Kx[2], conv_CHin[2], conv_CHout[2], conv_wt_scale[2],conv_wt_bit[2]);
	Load_Weight_From_bin_File(conv2_W, "Vgg16_weight/conv2_8b_quant_s10.bin");

	struct Mapped_Weight* conv3_W = Malloc_Weight(conv_Ky[3], conv_Kx[3], conv_CHin[3], conv_CHout[3], conv_wt_scale[3],conv_wt_bit[3]);
	Load_Weight_From_bin_File(conv3_W, "Vgg16_weight/conv3_8b_quant_s11.bin");
	
	struct Mapped_Weight* conv4_W = Malloc_Weight(conv_Ky[4], conv_Kx[4], conv_CHin[4], conv_CHout[4], conv_wt_scale[4],conv_wt_bit[4]);
	Load_Weight_From_bin_File(conv4_W, "Vgg16_weight/conv4_8b_quant_s11.bin");
	
	struct Mapped_Weight* conv5_W = Malloc_Weight(conv_Ky[5], conv_Kx[5], conv_CHin[5], conv_CHout[5], conv_wt_scale[5],conv_wt_bit[5]);
	Load_Weight_From_bin_File(conv5_W, "Vgg16_weight/conv5_4b_quant_s8.bin");
	
	struct Mapped_Weight* conv6_W = Malloc_Weight(conv_Ky[6], conv_Kx[6], conv_CHin[6], conv_CHout[6], conv_wt_scale[6],conv_wt_bit[6]);
	Load_Weight_From_bin_File(conv6_W, "Vgg16_weight/conv6_8b_quant_s12.bin");
	
	struct Mapped_Weight* conv7_W = Malloc_Weight(conv_Ky[7], conv_Kx[7], conv_CHin[7], conv_CHout[7], conv_wt_scale[7],conv_wt_bit[7]);
	Load_Weight_From_bin_File(conv7_W, "Vgg16_weight/conv7_8b_quant_s12.bin");
	
	struct Mapped_Weight* conv8_W = Malloc_Weight(conv_Ky[8], conv_Kx[8], conv_CHin[8], conv_CHout[8], conv_wt_scale[8],conv_wt_bit[8]);
	Load_Weight_From_bin_File(conv8_W, "Vgg16_weight/conv8_4b_quant_s8.bin");
	
	struct Mapped_Weight* conv9_W = Malloc_Weight(conv_Ky[9], conv_Kx[9], conv_CHin[9], conv_CHout[9], conv_wt_scale[9],conv_wt_bit[9]);
	Load_Weight_From_bin_File(conv9_W, "Vgg16_weight/conv9_8b_quant_s12.bin");
	
	struct Mapped_Weight* conv10_W = Malloc_Weight(conv_Ky[10], conv_Kx[10], conv_CHin[10], conv_CHout[10], conv_wt_scale[10],conv_wt_bit[10]);
	Load_Weight_From_bin_File(conv10_W, "Vgg16_weight/conv10_8b_quant_s12.bin");
	
	struct Mapped_Weight* conv11_W = Malloc_Weight(conv_Ky[11], conv_Kx[11], conv_CHin[11], conv_CHout[11], conv_wt_scale[11],conv_wt_bit[11]);
	Load_Weight_From_bin_File(conv11_W, "Vgg16_weight/conv11_8b_quant_s12.bin");
	
	struct Mapped_Weight* conv12_W = Malloc_Weight(conv_Ky[12], conv_Kx[12], conv_CHin[12], conv_CHout[12], conv_wt_scale[12],conv_wt_bit[12]);
	Load_Weight_From_bin_File(conv12_W, "Vgg16_weight/conv12_8b_quant_s12.bin");
	
	struct Mapped_Weight* conv13_W = Malloc_Weight(conv_Ky[13], conv_Kx[13], conv_CHin[13], conv_CHout[13], conv_wt_scale[13],conv_wt_bit[13]);
	Load_Weight_From_bin_File(conv13_W, "Vgg16_weight/conv13_4b_quant_s11.bin");
	
	struct Mapped_Weight* conv14_W = Malloc_Weight(conv_Ky[14], conv_Kx[14], conv_CHin[14], conv_CHout[14], conv_wt_scale[14],conv_wt_bit[14]);
	Load_Weight_From_bin_File(conv14_W, "Vgg16_weight/conv14_8b_quant_s14.bin");
	
	struct Mapped_Weight* conv15_W = Malloc_Weight(conv_Ky[15], conv_Kx[15], conv_CHin[15], conv_CHout[15], conv_wt_scale[15],conv_wt_bit[15]);
	Load_Weight_From_bin_File(conv15_W, "Vgg16_weight/conv15_8b_quant_s13.bin");
	
	printf("load wt Done\n");
	
	Load_Feature_From_bin_File(image, "Vgg16_result_from_C/parallel_image_8b_Tout32.bin");
	printf("load image done! \n");	

	
	printf("*************\n");
	printf("conv0 start\n");
	clock_t start_conv0, end_conv0;
	start_conv0=clock();
    for(int i=0; i<1000; i++)
    {
		FPGA_Conv(conv0_cfg, 1, image, conv0_W, conv0_relu_out);
	}
    end_conv0=clock();
    printf("conv0 Finish, run time is %ld us \n",(end_conv0-start_conv0)/1000);
	
	printf("*************\n");
	printf("conv1 start\n");
	clock_t start_conv1, end_conv1;
	start_conv1=clock();
    for(int i=0; i<1000; i++)
    {
		FPGA_Conv(conv1_cfg, 1, conv0_relu_out, conv1_W, conv1_relu_out);
	}
    end_conv1=clock();
    printf("conv1 Finish, run time is %ld us \n",(end_conv1-start_conv1)/1000);
	
	printf("*************\n");
	printf("pool0 start\n");
	clock_t start_pool0, end_pool0;
    start_pool0=clock();
	for(int i=0; i<1000; i++)
    {
		FPGA_Pool(pool0_cfg, 1, conv1_relu_out, pool0_out);
	}
    end_pool0=clock();
    printf("pool0 Finish, run time is %ld us \n",(end_pool0-start_pool0)/1000);
	
	
	
	printf("*************\n");
	printf("conv2 start\n");
	clock_t start_conv2, end_conv2;
    start_conv2=clock();
	for(int i=0; i<1000; i++)
    {
		FPGA_Conv(conv2_cfg, 1, pool0_out, conv2_W, conv2_relu_out);
	}
    end_conv2=clock();
    printf("conv2 Finish, run time is %ld us \n",(end_conv2-start_conv2)/1000);
	
	printf("*************\n");
	printf("conv3 start\n");
	clock_t start_conv3, end_conv3;
    start_conv3=clock();
	for(int i=0; i<1000; i++)
    {
		FPGA_Conv(conv3_cfg, 1, conv2_relu_out, conv3_W, conv3_relu_out);
	}
    end_conv3=clock();
    printf("conv3 Finish, run time is %ld us \n",(end_conv3-start_conv3)/1000);
	
	printf("*************\n");
	printf("pool1 start\n");
	clock_t start_pool1, end_pool1;
    start_pool1=clock();
	for(int i=0; i<1000; i++)
    {
		FPGA_Pool(pool1_cfg, 1, conv3_relu_out, pool1_out);
	}
    end_pool1=clock();
    printf("pool1 Finish, run time is %ld us \n",(end_pool1-start_pool1)/1000);
	
	
	
	printf("*************\n");
	printf("conv4 start\n");
	clock_t start_conv4, end_conv4;
    start_conv4=clock();
	for(int i=0; i<1000; i++)
    {
		FPGA_Conv(conv4_cfg, 1, pool1_out, conv4_W, conv4_relu_out);
	}
    end_conv4=clock();
    printf("conv4 Finish, run time is %ld us \n",(end_conv4-start_conv4)/1000);
	
	printf("*************\n");
	printf("conv5 start\n");
	clock_t start_conv5, end_conv5;
    start_conv5=clock();
	for(int i=0; i<1000; i++)
    {
		FPGA_Conv(conv5_cfg, 1, conv4_relu_out, conv5_W, conv5_relu_out);
	}
    end_conv5=clock();
    printf("conv5 Finish, run time is %ld us \n",(end_conv5-start_conv5)/1000);
	
	printf("*************\n");
	printf("conv6 start\n");
	clock_t start_conv6, end_conv6;
    start_conv6=clock();
	for(int i=0; i<1000; i++)
    {
		FPGA_Conv(conv6_cfg, 1, conv5_relu_out, conv6_W, conv6_relu_out);
	}
    end_conv6=clock();
    printf("conv6 Finish, run time is %ld us \n",(end_conv6-start_conv6)/1000);
	
	printf("*************\n");
	printf("pool2 start\n");
	clock_t start_pool2, end_pool2;
    start_pool2=clock();
	for(int i=0; i<1000; i++)
    {
		FPGA_Pool(pool2_cfg, 1, conv6_relu_out, pool2_out);
	}
    end_pool2=clock();
    printf("pool2 Finish, run time is %ld us \n",(end_pool2-start_pool2)/1000);
	
	
	
	printf("*************\n");
	printf("conv7 start\n");
	clock_t start_conv7, end_conv7;
    start_conv7=clock();
	for(int i=0; i<1000; i++)
    {
		FPGA_Conv(conv7_cfg, 1, pool2_out, conv7_W, conv7_relu_out);
	}
    end_conv7=clock();
    printf("conv7 Finish, run time is %ld us \n",(end_conv7-start_conv7)/1000);
	
	printf("*************\n");
	printf("conv8 start\n");
	clock_t start_conv8, end_conv8;
    start_conv8=clock();
	for(int i=0; i<1000; i++)
    {
		FPGA_Conv(conv8_cfg, 1, conv7_relu_out, conv8_W, conv8_relu_out);
	}
    end_conv8=clock();
    printf("conv8 Finish, run time is %ld us \n",(end_conv8-start_conv8)/1000);
	
	
	printf("*************\n");
	printf("conv9 start\n");
	clock_t start_conv9, end_conv9;
    start_conv9=clock();
	for(int i=0; i<1000; i++)
    {
		FPGA_Conv(conv9_cfg, 1, conv8_relu_out, conv9_W, conv9_relu_out);
	}
    end_conv9=clock();
    printf("conv9 Finish, run time is %ld us \n",(end_conv9-start_conv9)/1000);
	
	printf("*************\n");
	printf("pool3 start\n");
	clock_t start_pool3, end_pool3;
    start_pool3=clock();
	for(int i=0; i<1000; i++)
    {
		FPGA_Pool(pool3_cfg, 1, conv9_relu_out, pool3_out);
	}
    end_pool3=clock();
    printf("pool3 Finish, run time is %ld us \n",(end_pool3-start_pool3)/1000);
	
	
	
	printf("*************\n");
	printf("conv10 start\n");
	clock_t start_conv10, end_conv10;
    start_conv10=clock();
	for(int i=0; i<1000; i++)
    {
		FPGA_Conv(conv10_cfg, 1, pool3_out, conv10_W, conv10_relu_out);
	}
    end_conv10=clock();
    printf("conv10 Finish, run time is %ld us \n",(end_conv10-start_conv10)/1000);
	
	printf("*************\n");
	printf("conv11 start\n");
	clock_t start_conv11, end_conv11;
    start_conv11=clock();
	for(int i=0; i<1000; i++)
    {
		FPGA_Conv(conv11_cfg, 1, conv10_relu_out, conv11_W, conv11_relu_out);
	}
    end_conv11=clock();
    printf("conv11 Finish, run time is %ld us \n",(end_conv11-start_conv11)/1000);
	
	printf("*************\n");
	printf("conv12 start\n");
	clock_t start_conv12, end_conv12;
    start_conv12=clock();
	for(int i=0; i<1000; i++)
    {
		FPGA_Conv(conv12_cfg, 1, conv11_relu_out, conv12_W, conv12_relu_out);
	}
    end_conv12=clock();
    printf("conv12 Finish, run time is %ld us \n",(end_conv12-start_conv12)/1000);
	
	printf("*************\n");
	printf("pool4 start\n");
	clock_t start_pool4, end_pool4;
    start_pool4=clock();
	for(int i=0; i<1000; i++)
    {
		FPGA_Pool(pool4_cfg, 1, conv12_relu_out, pool4_out);
	}
    end_pool4=clock();
    printf("pool4 Finish, run time is %ld us \n",(end_pool4-start_pool4)/1000);
	
	
	
	printf("*************\n");
	printf("conv13 start\n");
	clock_t start_conv13, end_conv13;
    start_conv13=clock();
	for(int i=0; i<1000; i++)
    {
		FPGA_Conv(conv13_cfg, 1, pool4_out, conv13_W, conv13_relu_out);
	}
    end_conv13=clock();
    printf("conv13 Finish, run time is %ld us \n",(end_conv13-start_conv13)/1000);
	
	printf("*************\n");
	printf("conv14 start\n");
	clock_t start_conv14, end_conv14;
    start_conv14=clock();
	for(int i=0; i<1000; i++)
    {
		FPGA_Conv(conv14_cfg, 1, conv13_relu_out, conv14_W, conv14_relu_out);
	}
    end_conv14=clock();
    printf("conv14 Finish, run time is %ld us \n",(end_conv14-start_conv14)/1000);
	
	
	printf("*************\n");
	printf("conv15 start\n");
	clock_t start_conv15, end_conv15;
    start_conv15=clock();
	for(int i=0; i<1000; i++)
    {
		FPGA_Conv(conv15_cfg, 1, conv14_relu_out, conv15_W, conv15_relu_out);
	}
    end_conv15=clock();
    printf("conv15 Finish, run time is %ld us \n",(end_conv15-start_conv15)/1000);
	
	
	printf("\n");

}
