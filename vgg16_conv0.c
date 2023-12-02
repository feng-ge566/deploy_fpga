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

	struct Mapped_Feature* image = Malloc_Feature(conv_Hin[0], conv_Win[0], conv_CHin[0], conv_in_scale[0],conv_out_scale[0],conv_in_bit[0]);
	struct Mapped_Feature* conv0_relu_out = Malloc_Feature(conv_Hout[0], conv_Wout[0], conv_CHout[0], conv_out_scale[0],conv_out_scale[0],conv_out_bit[0]);
	printf("Malloc Memory Done\n");

	struct Mapped_Weight* conv0_W = Malloc_Weight(conv_Ky[0], conv_Kx[0], conv_CHin[0], conv_CHout[0], conv_wt_scale[0],conv_wt_bit[0]);
	Load_Weight_From_bin_File(conv0_W, "test/wt_8b_all1_Tout32.bin");
	printf("load conv0_wt Done\n");
	
	Load_Feature_From_bin_File(image, "test/dat_8b_all1_Tout32.bin");
	printf("load image done! \n");	

	printf("\n the conv0_in[0][0][0]  is: %6d",  *Get_Element(image, 0, 0, 0));
	printf("\n the conv0_in[0][0][1]  is: %6d",  *Get_Element(image, 0, 0, 1));
	printf("\n the conv0_in[0][0][2]  is: %6d",  *Get_Element(image, 0, 0, 2));
	printf("\n the conv0_in[0][0][3]  is: %6d",  *Get_Element(image, 0, 0, 3));
	printf("\n the conv0_in[0][0][4]  is: %6d",  *Get_Element(image, 0, 0, 4));
	printf("\n the conv0_in[0][0][5]  is: %6d",  *Get_Element(image, 0, 0, 5));
	printf("\n the conv0_in[0][0][6]  is: %6d",  *Get_Element(image, 0, 0, 6));
	printf("\n the conv0_in[0][0][7]  is: %6d",  *Get_Element(image, 0, 0, 7));
	printf("\n the conv0_in[0][0][8]  is: %6d",  *Get_Element(image, 0, 0, 8));
	printf("\n the conv0_in[0][0][9]  is: %6d",  *Get_Element(image, 0, 0, 9));
	printf("\n the conv0_in[0][0][10] is: %6d", *Get_Element(image, 0, 0, 10));
	printf("\n the conv0_in[0][0][11] is: %6d", *Get_Element(image, 0, 0, 11));
	printf("\n the conv0_in[0][0][12] is: %6d", *Get_Element(image, 0, 0, 12));
	printf("\n the conv0_in[0][0][13] is: %6d", *Get_Element(image, 0, 0, 13));
	printf("\n the conv0_in[0][0][14] is: %6d", *Get_Element(image, 0, 0, 14));
	printf("\n the conv0_in[0][0][15] is: %6d", *Get_Element(image, 0, 0, 15));


	printf("\n the conv0_in[0][1][0]  is: %6d",  *Get_Element(image, 0, 1, 0));
	printf("\n the conv0_in[0][1][1]  is: %6d",  *Get_Element(image, 0, 1, 1));
	printf("\n the conv0_in[0][1][2]  is: %6d",  *Get_Element(image, 0, 1, 2));
	printf("\n the conv0_in[0][1][3]  is: %6d",  *Get_Element(image, 0, 1, 3));
	printf("\n the conv0_in[0][1][4]  is: %6d",  *Get_Element(image, 0, 1, 4));
	printf("\n the conv0_in[0][1][5]  is: %6d",  *Get_Element(image, 0, 1, 5));
	printf("\n the conv0_in[0][1][6]  is: %6d",  *Get_Element(image, 0, 1, 6));
	printf("\n the conv0_in[0][1][7]  is: %6d",  *Get_Element(image, 0, 1, 7));
	printf("\n the conv0_in[0][1][8]  is: %6d",  *Get_Element(image, 0, 1, 8));
	printf("\n the conv0_in[0][1][9]  is: %6d",  *Get_Element(image, 0, 1, 9));
	printf("\n the conv0_in[0][1][10] is: %6d", *Get_Element(image, 0, 1, 10));
	printf("\n the conv0_in[0][1][11] is: %6d", *Get_Element(image, 0, 1, 11));
	printf("\n the conv0_in[0][1][12] is: %6d", *Get_Element(image, 0, 1, 12));
	printf("\n the conv0_in[0][1][13] is: %6d", *Get_Element(image, 0, 1, 13));
	printf("\n the conv0_in[0][1][14] is: %6d", *Get_Element(image, 0, 1, 14));
	printf("\n the conv0_in[0][1][15] is: %6d", *Get_Element(image, 0, 1, 15));

	printf("\n the conv0_in[0][2][0]  is: %6d",  *Get_Element(image, 0, 2, 0));
	printf("\n the conv0_in[0][2][1]  is: %6d",  *Get_Element(image, 0, 2, 1));
	printf("\n the conv0_in[0][2][2]  is: %6d",  *Get_Element(image, 0, 2, 2));
	printf("\n the conv0_in[0][2][3]  is: %6d",  *Get_Element(image, 0, 2, 3));
	printf("\n the conv0_in[0][2][4]  is: %6d",  *Get_Element(image, 0, 2, 4));
	printf("\n the conv0_in[0][2][5]  is: %6d",  *Get_Element(image, 0, 2, 5));
	printf("\n the conv0_in[0][2][6]  is: %6d",  *Get_Element(image, 0, 2, 6));
	printf("\n the conv0_in[0][2][7]  is: %6d",  *Get_Element(image, 0, 2, 7));
	printf("\n the conv0_in[0][2][8]  is: %6d",  *Get_Element(image, 0, 2, 8));
	printf("\n the conv0_in[0][2][9]  is: %6d",  *Get_Element(image, 0, 2, 9));
	printf("\n the conv0_in[0][2][10] is: %6d", *Get_Element(image, 0, 2, 10));
	printf("\n the conv0_in[0][2][11] is: %6d", *Get_Element(image, 0, 2, 11));
	printf("\n the conv0_in[0][2][12] is: %6d", *Get_Element(image, 0, 2, 12));
	printf("\n the conv0_in[0][2][13] is: %6d", *Get_Element(image, 0, 2, 13));
	printf("\n the conv0_in[0][2][14] is: %6d", *Get_Element(image, 0, 2, 14));
	printf("\n the conv0_in[0][2][15] is: %6d", *Get_Element(image, 0, 2, 15));

	printf("\n the conv0_in[0][3][0]  is: %6d",  *Get_Element(image, 0, 3, 0));
	printf("\n the conv0_in[0][3][1]  is: %6d",  *Get_Element(image, 0, 3, 1));
	printf("\n the conv0_in[0][3][2]  is: %6d",  *Get_Element(image, 0, 3, 2));
	printf("\n the conv0_in[0][3][3]  is: %6d",  *Get_Element(image, 0, 3, 3));
	printf("\n the conv0_in[0][3][4]  is: %6d",  *Get_Element(image, 0, 3, 4));
	printf("\n the conv0_in[0][3][5]  is: %6d",  *Get_Element(image, 0, 3, 5));
	printf("\n the conv0_in[0][3][6]  is: %6d",  *Get_Element(image, 0, 3, 6));
	printf("\n the conv0_in[0][3][7]  is: %6d",  *Get_Element(image, 0, 3, 7));
	printf("\n the conv0_in[0][3][8]  is: %6d",  *Get_Element(image, 0, 3, 8));
	printf("\n the conv0_in[0][3][9]  is: %6d",  *Get_Element(image, 0, 3, 9));
	printf("\n the conv0_in[0][3][10] is: %6d", *Get_Element(image, 0, 3, 10));
	printf("\n the conv0_in[0][3][11] is: %6d", *Get_Element(image, 0, 3, 11));
	printf("\n the conv0_in[0][3][12] is: %6d", *Get_Element(image, 0, 3, 12));
	printf("\n the conv0_in[0][3][13] is: %6d", *Get_Element(image, 0, 3, 13));
	printf("\n the conv0_in[0][3][14] is: %6d", *Get_Element(image, 0, 3, 14));
	printf("\n the conv0_in[0][3][15] is: %6d", *Get_Element(image, 0, 3, 15));
	
	printf("\n the conv0_in[1][0][0]  is: %6d",  *Get_Element(image, 1, 0, 0));
	printf("\n the conv0_in[1][0][1]  is: %6d",  *Get_Element(image, 1, 0, 1));
	printf("\n the conv0_in[1][0][2]  is: %6d",  *Get_Element(image, 1, 0, 2));
	printf("\n the conv0_in[1][0][3]  is: %6d",  *Get_Element(image, 1, 0, 3));
	printf("\n the conv0_in[1][0][4]  is: %6d",  *Get_Element(image, 1, 0, 4));
	printf("\n the conv0_in[1][0][5]  is: %6d",  *Get_Element(image, 1, 0, 5));
	printf("\n the conv0_in[1][0][6]  is: %6d",  *Get_Element(image, 1, 0, 6));
	printf("\n the conv0_in[1][0][7]  is: %6d",  *Get_Element(image, 1, 0, 7));
	printf("\n the conv0_in[1][0][8]  is: %6d",  *Get_Element(image, 1, 0, 8));
	printf("\n the conv0_in[1][0][9]  is: %6d",  *Get_Element(image, 1, 0, 9));
	printf("\n the conv0_in[1][0][10] is: %6d", *Get_Element(image, 1, 0, 10));
	printf("\n the conv0_in[1][0][11] is: %6d", *Get_Element(image, 1, 0, 11));
	printf("\n the conv0_in[1][0][12] is: %6d", *Get_Element(image, 1, 0, 12));
	printf("\n the conv0_in[1][0][13] is: %6d", *Get_Element(image, 1, 0, 13));
	printf("\n the conv0_in[1][0][14] is: %6d", *Get_Element(image, 1, 0, 14));
	printf("\n the conv0_in[1][0][15] is: %6d", *Get_Element(image, 1, 0, 15));
	
	clock_t start_conv1, end_conv1;
	start_conv1=clock();
	//for(int count=0; count<1; count++)
	//{
		FPGA_Conv(conv0_cfg, 1, image, conv0_W, conv0_relu_out);
	//}
	end_conv1=clock();
	printf("\n conv1 Finish, run time is %ld ms \n",(end_conv1-start_conv1)/1000);
	Save_Feature_To_bin_File(conv0_relu_out, "test/conv0_dat1_wt1.bin");
	
	printf("\n the conv0_relu_out[0][0][0]  is: %6d",  *Get_Element(conv0_relu_out, 0, 0, 0));
	printf("\n the conv0_relu_out[0][0][1]  is: %6d",  *Get_Element(conv0_relu_out, 0, 0, 1));
	printf("\n the conv0_relu_out[0][0][2]  is: %6d",  *Get_Element(conv0_relu_out, 0, 0, 2));
	printf("\n the conv0_relu_out[0][0][3]  is: %6d",  *Get_Element(conv0_relu_out, 0, 0, 3));
	printf("\n the conv0_relu_out[0][0][4]  is: %6d",  *Get_Element(conv0_relu_out, 0, 0, 4));
	printf("\n the conv0_relu_out[0][0][5]  is: %6d",  *Get_Element(conv0_relu_out, 0, 0, 5));
	printf("\n the conv0_relu_out[0][0][6]  is: %6d",  *Get_Element(conv0_relu_out, 0, 0, 6));
	printf("\n the conv0_relu_out[0][0][7]  is: %6d",  *Get_Element(conv0_relu_out, 0, 0, 7));
	printf("\n the conv0_relu_out[0][0][8]  is: %6d",  *Get_Element(conv0_relu_out, 0, 0, 8));
	printf("\n the conv0_relu_out[0][0][9]  is: %6d",  *Get_Element(conv0_relu_out, 0, 0, 9));
	printf("\n the conv0_relu_out[0][0][10] is: %6d", *Get_Element(conv0_relu_out, 0, 0, 10));
	printf("\n the conv0_relu_out[0][0][11] is: %6d", *Get_Element(conv0_relu_out, 0, 0, 11));
	printf("\n the conv0_relu_out[0][0][12] is: %6d", *Get_Element(conv0_relu_out, 0, 0, 12));
	printf("\n the conv0_relu_out[0][0][13] is: %6d", *Get_Element(conv0_relu_out, 0, 0, 13));
	printf("\n the conv0_relu_out[0][0][14] is: %6d", *Get_Element(conv0_relu_out, 0, 0, 14));
	printf("\n the conv0_relu_out[0][0][15] is: %6d", *Get_Element(conv0_relu_out, 0, 0, 15));
	
	printf("\n the conv0_relu_out[0][1][0]  is: %6d",  *Get_Element(conv0_relu_out, 0, 1, 0));
	printf("\n the conv0_relu_out[0][1][1]  is: %6d",  *Get_Element(conv0_relu_out, 0, 1, 1));
	printf("\n the conv0_relu_out[0][1][2]  is: %6d",  *Get_Element(conv0_relu_out, 0, 1, 2));
	printf("\n the conv0_relu_out[0][1][3]  is: %6d",  *Get_Element(conv0_relu_out, 0, 1, 3));
	printf("\n the conv0_relu_out[0][1][4]  is: %6d",  *Get_Element(conv0_relu_out, 0, 1, 4));
	printf("\n the conv0_relu_out[0][1][5]  is: %6d",  *Get_Element(conv0_relu_out, 0, 1, 5));
	printf("\n the conv0_relu_out[0][1][6]  is: %6d",  *Get_Element(conv0_relu_out, 0, 1, 6));
	printf("\n the conv0_relu_out[0][1][7]  is: %6d",  *Get_Element(conv0_relu_out, 0, 1, 7));
	printf("\n the conv0_relu_out[0][1][8]  is: %6d",  *Get_Element(conv0_relu_out, 0, 1, 8));
	printf("\n the conv0_relu_out[0][1][9]  is: %6d",  *Get_Element(conv0_relu_out, 0, 1, 9));
	printf("\n the conv0_relu_out[0][1][10] is: %6d", *Get_Element(conv0_relu_out, 0, 1, 10));
	printf("\n the conv0_relu_out[0][1][11] is: %6d", *Get_Element(conv0_relu_out, 0, 1, 11));
	printf("\n the conv0_relu_out[0][1][12] is: %6d", *Get_Element(conv0_relu_out, 0, 1, 12));
	printf("\n the conv0_relu_out[0][1][13] is: %6d", *Get_Element(conv0_relu_out, 0, 1, 13));
	printf("\n the conv0_relu_out[0][1][14] is: %6d", *Get_Element(conv0_relu_out, 0, 1, 14));
	printf("\n the conv0_relu_out[0][1][15] is: %6d", *Get_Element(conv0_relu_out, 0, 1, 15));

	printf("\n the conv0_relu_out[0][2][0]  is: %6d",  *Get_Element(conv0_relu_out, 0, 2, 0));
	printf("\n the conv0_relu_out[0][2][1]  is: %6d",  *Get_Element(conv0_relu_out, 0, 2, 1));
	printf("\n the conv0_relu_out[0][2][2]  is: %6d",  *Get_Element(conv0_relu_out, 0, 2, 2));
	printf("\n the conv0_relu_out[0][2][3]  is: %6d",  *Get_Element(conv0_relu_out, 0, 2, 3));
	printf("\n the conv0_relu_out[0][2][4]  is: %6d",  *Get_Element(conv0_relu_out, 0, 2, 4));
	printf("\n the conv0_relu_out[0][2][5]  is: %6d",  *Get_Element(conv0_relu_out, 0, 2, 5));
	printf("\n the conv0_relu_out[0][2][6]  is: %6d",  *Get_Element(conv0_relu_out, 0, 2, 6));
	printf("\n the conv0_relu_out[0][2][7]  is: %6d",  *Get_Element(conv0_relu_out, 0, 2, 7));
	printf("\n the conv0_relu_out[0][2][8]  is: %6d",  *Get_Element(conv0_relu_out, 0, 2, 8));
	printf("\n the conv0_relu_out[0][2][9]  is: %6d",  *Get_Element(conv0_relu_out, 0, 2, 9));
	printf("\n the conv0_relu_out[0][2][10] is: %6d", *Get_Element(conv0_relu_out, 0, 2, 10));
	printf("\n the conv0_relu_out[0][2][11] is: %6d", *Get_Element(conv0_relu_out, 0, 2, 11));
	printf("\n the conv0_relu_out[0][2][12] is: %6d", *Get_Element(conv0_relu_out, 0, 2, 12));
	printf("\n the conv0_relu_out[0][2][13] is: %6d", *Get_Element(conv0_relu_out, 0, 2, 13));
	printf("\n the conv0_relu_out[0][2][14] is: %6d", *Get_Element(conv0_relu_out, 0, 2, 14));
	printf("\n the conv0_relu_out[0][2][15] is: %6d", *Get_Element(conv0_relu_out, 0, 2, 15));
	
	printf("\n the conv0_relu_out[0][3][0]  is: %6d",  *Get_Element(conv0_relu_out, 0, 3, 0));
	printf("\n the conv0_relu_out[0][3][1]  is: %6d",  *Get_Element(conv0_relu_out, 0, 3, 1));
	printf("\n the conv0_relu_out[0][3][2]  is: %6d",  *Get_Element(conv0_relu_out, 0, 3, 2));
	printf("\n the conv0_relu_out[0][3][3]  is: %6d",  *Get_Element(conv0_relu_out, 0, 3, 3));
	printf("\n the conv0_relu_out[0][3][4]  is: %6d",  *Get_Element(conv0_relu_out, 0, 3, 4));
	printf("\n the conv0_relu_out[0][3][5]  is: %6d",  *Get_Element(conv0_relu_out, 0, 3, 5));
	printf("\n the conv0_relu_out[0][3][6]  is: %6d",  *Get_Element(conv0_relu_out, 0, 3, 6));
	printf("\n the conv0_relu_out[0][3][7]  is: %6d",  *Get_Element(conv0_relu_out, 0, 3, 7));
	printf("\n the conv0_relu_out[0][3][8]  is: %6d",  *Get_Element(conv0_relu_out, 0, 3, 8));
	printf("\n the conv0_relu_out[0][3][9]  is: %6d",  *Get_Element(conv0_relu_out, 0, 3, 9));
	printf("\n the conv0_relu_out[0][3][10] is: %6d", *Get_Element(conv0_relu_out, 0, 3, 10));
	printf("\n the conv0_relu_out[0][3][11] is: %6d", *Get_Element(conv0_relu_out, 0, 3, 11));
	printf("\n the conv0_relu_out[0][3][12] is: %6d", *Get_Element(conv0_relu_out, 0, 3, 12));
	printf("\n the conv0_relu_out[0][3][13] is: %6d", *Get_Element(conv0_relu_out, 0, 3, 13));
	printf("\n the conv0_relu_out[0][3][14] is: %6d", *Get_Element(conv0_relu_out, 0, 3, 14));
	printf("\n the conv0_relu_out[0][3][15] is: %6d", *Get_Element(conv0_relu_out, 0, 3, 15));
	
	printf("\n the conv0_relu_out[1][1][0]  is: %6d",  *Get_Element(conv0_relu_out, 1, 1, 0));
	printf("\n the conv0_relu_out[1][1][1]  is: %6d",  *Get_Element(conv0_relu_out, 1, 1, 1));
	printf("\n the conv0_relu_out[1][1][2]  is: %6d",  *Get_Element(conv0_relu_out, 1, 1, 2));
	printf("\n the conv0_relu_out[1][1][3]  is: %6d",  *Get_Element(conv0_relu_out, 1, 1, 3));
	printf("\n the conv0_relu_out[1][1][4]  is: %6d",  *Get_Element(conv0_relu_out, 1, 1, 4));
	printf("\n the conv0_relu_out[1][1][5]  is: %6d",  *Get_Element(conv0_relu_out, 1, 1, 5));
	printf("\n the conv0_relu_out[1][1][6]  is: %6d",  *Get_Element(conv0_relu_out, 1, 1, 6));
	printf("\n the conv0_relu_out[1][1][7]  is: %6d",  *Get_Element(conv0_relu_out, 1, 1, 7));
	printf("\n the conv0_relu_out[1][1][8]  is: %6d",  *Get_Element(conv0_relu_out, 1, 1, 8));
	printf("\n the conv0_relu_out[1][1][9]  is: %6d",  *Get_Element(conv0_relu_out, 1, 1, 9));
	printf("\n the conv0_relu_out[1][1][10] is: %6d", *Get_Element(conv0_relu_out, 1, 1, 10));
	printf("\n the conv0_relu_out[1][1][11] is: %6d", *Get_Element(conv0_relu_out, 1, 1, 11));
	printf("\n the conv0_relu_out[1][1][12] is: %6d", *Get_Element(conv0_relu_out, 1, 1, 12));
	printf("\n the conv0_relu_out[1][1][13] is: %6d", *Get_Element(conv0_relu_out, 1, 1, 13));
	printf("\n the conv0_relu_out[1][1][14] is: %6d", *Get_Element(conv0_relu_out, 1, 1, 14));
	printf("\n the conv0_relu_out[1][1][15] is: %6d", *Get_Element(conv0_relu_out, 1, 1, 15));
	
	printf("\n");

}
