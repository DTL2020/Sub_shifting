// Asm_test01.cpp : Defines the entry point for the console application.
//
#include "pch.h"
#include <immintrin.h> // MS version of immintrin.h covers AVX, AVX2 and FMA3
#include <math.h>

unsigned char RefPlane[100 * 100];
unsigned char CurrBlock[16 * 16];

unsigned int nSrcPitch[3] = { 16,16,16 };
unsigned int nRefPitch[3] = { 100,100,100 };

const int iBlockSizeX = 8;
const int iBlockSizeY = 8;

const int iBlockSizeX_UV = 4;
const int iBlockSizeY_UV = 4;

const int iKS = 8;
int iKS_d2 = iKS / 2;

float CurrBlockShiftH[iBlockSizeX * (iBlockSizeY + iKS)];
unsigned char CurrBlockShiftedHV[iBlockSizeX * iBlockSizeY];

float CurrBlockShiftH_avx2[iBlockSizeX * (iBlockSizeY + iKS)];
float CurrBlockShiftH_UV_avx2[iBlockSizeX_UV * (iBlockSizeY_UV + iKS)];
unsigned char CurrBlockShiftedHV_avx2[iBlockSizeX * iBlockSizeY];


float fPi = 3.14159265f;
float fFreqH = 0.25f;
float fFreqV = 0.25f;
//float fPelShiftH = 0.25f;// 0.5f;
//float fPelShiftV = 0.25f;// 0.5f;

int iPelH = 1;
int iPelV = 1;

float fKernelH_01[iKS];
float fKernelH_10[iKS];
float fKernelH_11[iKS];

float fKernelV_01[iKS];
float fKernelV_10[iKS];
float fKernelV_11[iKS];


float fSinc(float x)
{
	x = fabsf(x);

	if (x > 0.000001f)
	{
		return sinf(x) / x;
	}
	else return 1.0f;
}

void CalcShiftKernel(float *fKernel, float fPelShift, int iKS)
{
	for (int i = 0; i < iKS; i++)
	{
		float fArg = (float)(i - iKS_d2)*fPi + fPelShift;
		fKernel[i] = fSinc(fArg);
	}

	float fSum = 0.0f;
	for (int i = 0; i < iKS; i++)
	{
		fSum += fKernel[i];
	}

	for (int i = 0; i < iKS; i++)
	{
		fKernel[i] /= fSum;
	}
}

void SubShiftBlock_C(float *fKernelH, float *fKernelV)
{
	if (fKernelH != 0)
	{
		for (int j = 0; j < (iBlockSizeY + iKS); j++)
		{
			for (int i = 0; i < iBlockSizeX; i++)
			{
				float fOut = 0.0f;

				for (int k = 0; k < iKS; k++)
				{
					float fSample = (float)CurrBlock[j * nSrcPitch[0] + i + k];
					fOut += fSample * fKernelH[k];
				}

				CurrBlockShiftH[j * iBlockSizeX + i] = fOut;
			}
		}
	}
	else // copy to CurrBlockShiftH temp buf
	{
		for (int j = 0; j < (iBlockSizeY + iKS); j++)
		{
			for (int i = 0; i < iBlockSizeX; i++)
			{
				CurrBlockShiftH[j * iBlockSizeX + i] = (float)CurrBlock[j * nSrcPitch[0] + i + iKS_d2];;
			}
		}
	}

	if (fKernelV != 0)
	{
		// V shift
		for (int i = 0; i < iBlockSizeX; i++)
		{
			for (int j = 0; j < iBlockSizeY; j++)
			{
				float fOut = 0.0f;

				for (int k = 0; k < iKS; k++)
				{
					float fSample = CurrBlockShiftH[(j + k) * iBlockSizeX + i];
					fOut += fSample * fKernelV[k];
				}

				fOut += 0.5f;

				if (fOut > 255.0f) fOut = 255.0f;
				if (fOut < 0.0f) fOut = 0.0f;

				CurrBlockShiftedHV[j * iBlockSizeX + i] = (unsigned char)(fOut);
			}
		}
	}
	else // copy to out buf
	{
		for (int i = 0; i < iBlockSizeX; i++)
		{
			for (int j = 0; j < iBlockSizeY; j++)
			{
				float fOut = CurrBlockShiftH[(j + iKS_d2) * iBlockSizeX + i] + 0.5f;
				if (fOut > 255.0f) fOut = 255.0f;
				if (fOut < 0.0f) fOut = 0.0f;
				CurrBlockShiftedHV[j * iBlockSizeX + i] = (unsigned char)(fOut);
			}
		}
	}
}


void SubShiftBlock8x8_KS8_avx2(float *fKernelH, float *fKernelV)
{
	const int iSrcStride = 16;
	const int iHShiftedStride = 8;
	const int iHVShiftedStride = 8;

	if (fKernelH != 0)
	{
		unsigned char *pSrc = CurrBlock;
		float *pDst = CurrBlockShiftH_avx2;
		__m256 ymm_Krn;
		__m256i ymm_perm_rot_1_float_left = _mm256_set_epi32(0, 7, 6, 5, 4, 3, 2, 1);

		__m256 ymm0_row0_0f, ymm1_row0_1f;
		__m256 ymm2_row1_0f, ymm3_row1_1f;
		__m256 ymm4_row2_0f, ymm5_row2_1f;
		__m256 ymm6_row3_0f, ymm7_row3_1f;


		for (int y = 0; y < 4; y++) // 4 groups of 4 rows
		{
			__m256 ymm8_out_row0 = _mm256_setzero_ps();
			__m256 ymm9_out_row1 = _mm256_setzero_ps();
			__m256 ymm10_out_row2 = _mm256_setzero_ps();
			__m256 ymm11_out_row3 = _mm256_setzero_ps();

			ymm0_row0_0f = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadu_si64(pSrc)));
			ymm1_row0_1f = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadu_si64(pSrc + 8)));

			ymm2_row1_0f = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadu_si64(pSrc + iSrcStride)));
			ymm3_row1_1f = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadu_si64(pSrc + iSrcStride + 8)));

			ymm4_row2_0f = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadu_si64(pSrc + iSrcStride * 2)));
			ymm5_row2_1f = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadu_si64(pSrc + iSrcStride * 2 + 8)));

			ymm6_row3_0f = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadu_si64(pSrc + iSrcStride * 3)));
			ymm7_row3_1f = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadu_si64(pSrc + iSrcStride * 3 + 8)));

			ymm_Krn = _mm256_broadcast_ss(fKernelH);

			// 1 of 8
			ymm8_out_row0 = _mm256_fmadd_ps(ymm0_row0_0f, ymm_Krn, ymm8_out_row0);
			ymm9_out_row1 = _mm256_fmadd_ps(ymm2_row1_0f, ymm_Krn, ymm9_out_row1);
			ymm10_out_row2 = _mm256_fmadd_ps(ymm4_row2_0f, ymm_Krn, ymm10_out_row2);
			ymm11_out_row3 = _mm256_fmadd_ps(ymm6_row3_0f, ymm_Krn, ymm11_out_row3);

			for (int x = 1; x < 8; x++)
			{
				//shift 1 float sample
				ymm0_row0_0f = _mm256_permutevar8x32_ps(ymm0_row0_0f, ymm_perm_rot_1_float_left);
				ymm1_row0_1f = _mm256_permutevar8x32_ps(ymm1_row0_1f, ymm_perm_rot_1_float_left);
				ymm0_row0_0f = _mm256_blend_ps(ymm0_row0_0f, ymm1_row0_1f, 128);

				ymm2_row1_0f = _mm256_permutevar8x32_ps(ymm2_row1_0f, ymm_perm_rot_1_float_left);
				ymm3_row1_1f = _mm256_permutevar8x32_ps(ymm3_row1_1f, ymm_perm_rot_1_float_left);
				ymm2_row1_0f = _mm256_blend_ps(ymm2_row1_0f, ymm3_row1_1f, 128);

				ymm4_row2_0f = _mm256_permutevar8x32_ps(ymm4_row2_0f, ymm_perm_rot_1_float_left);
				ymm5_row2_1f = _mm256_permutevar8x32_ps(ymm5_row2_1f, ymm_perm_rot_1_float_left);
				ymm4_row2_0f = _mm256_blend_ps(ymm4_row2_0f, ymm5_row2_1f, 128);

				ymm6_row3_0f = _mm256_permutevar8x32_ps(ymm6_row3_0f, ymm_perm_rot_1_float_left);
				ymm7_row3_1f = _mm256_permutevar8x32_ps(ymm7_row3_1f, ymm_perm_rot_1_float_left);
				ymm6_row3_0f = _mm256_blend_ps(ymm6_row3_0f, ymm7_row3_1f, 128);

				ymm_Krn = _mm256_broadcast_ss(fKernelH + x);

				ymm8_out_row0 = _mm256_fmadd_ps(ymm0_row0_0f, ymm_Krn, ymm8_out_row0);
				ymm9_out_row1 = _mm256_fmadd_ps(ymm2_row1_0f, ymm_Krn, ymm9_out_row1);
				ymm10_out_row2 = _mm256_fmadd_ps(ymm4_row2_0f, ymm_Krn, ymm10_out_row2);
				ymm11_out_row3 = _mm256_fmadd_ps(ymm6_row3_0f, ymm_Krn, ymm11_out_row3);

			}

			_mm256_storeu_ps(pDst, ymm8_out_row0);
			_mm256_storeu_ps(pDst + 8, ymm9_out_row1);
			_mm256_storeu_ps(pDst + 16, ymm10_out_row2);
			_mm256_storeu_ps(pDst + 24, ymm11_out_row3);
			
			pSrc = pSrc + (iSrcStride * 4); // in bytes
			pDst = pDst + (iHShiftedStride * 4); // in float32

		}
	}
	else // copy to CurrBlockShiftH temp buf
	{
		unsigned char *pSrc = CurrBlock + 4; // iKS_d2
		float *pDst = CurrBlockShiftH_avx2;

		_mm256_storeu_ps(pDst, _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadu_si64(pSrc + 0))));
		_mm256_storeu_ps(pDst + iHShiftedStride * 1, _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadu_si64(pSrc + iSrcStride * 1)))); // may be use add iSrcStride each step ? need to check into disassember
		_mm256_storeu_ps(pDst + iHShiftedStride * 2, _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadu_si64(pSrc + iSrcStride * 2))));
		_mm256_storeu_ps(pDst + iHShiftedStride * 3, _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadu_si64(pSrc + iSrcStride * 3))));
		_mm256_storeu_ps(pDst + iHShiftedStride * 4, _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadu_si64(pSrc + iSrcStride * 4))));
		_mm256_storeu_ps(pDst + iHShiftedStride * 5, _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadu_si64(pSrc + iSrcStride * 5))));
		_mm256_storeu_ps(pDst + iHShiftedStride * 6, _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadu_si64(pSrc + iSrcStride * 6))));
		_mm256_storeu_ps(pDst + iHShiftedStride * 7, _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadu_si64(pSrc + iSrcStride * 7))));
		_mm256_storeu_ps(pDst + iHShiftedStride * 8, _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadu_si64(pSrc + iSrcStride * 8))));
		_mm256_storeu_ps(pDst + iHShiftedStride * 9, _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadu_si64(pSrc + iSrcStride * 9))));
		_mm256_storeu_ps(pDst + iHShiftedStride * 10, _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadu_si64(pSrc + iSrcStride * 10))));
		_mm256_storeu_ps(pDst + iHShiftedStride * 11, _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadu_si64(pSrc + iSrcStride * 11))));
		_mm256_storeu_ps(pDst + iHShiftedStride * 12, _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadu_si64(pSrc + iSrcStride * 12))));
		_mm256_storeu_ps(pDst + iHShiftedStride * 13, _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadu_si64(pSrc + iSrcStride * 13))));
		_mm256_storeu_ps(pDst + iHShiftedStride * 14, _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadu_si64(pSrc + iSrcStride * 14))));
		_mm256_storeu_ps(pDst + iHShiftedStride * 15, _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadu_si64(pSrc + iSrcStride * 15))));

	}

	if (fKernelV != 0)
	{
		// V shift
		float *pfSrc = CurrBlockShiftH_avx2;
		unsigned char *pucDst = CurrBlockShiftedHV_avx2;

		__m256 ymm0_out0, ymm1_out1, ymm2_out2, ymm3_out3, ymm4_out4, ymm5_out5, ymm6_out6, ymm7_out7;
		__m256 ymm8_Krn;

		float fZero = 0.0f;
		__m256 ymm9_MaxPS = _mm256_broadcast_ss(&fZero);

		float f255 = 255.0f;
		__m256 ymm10_MinPS = _mm256_broadcast_ss(&f255);

		__m256i ymm11_8bit_perm = _mm256_set_epi32(7, 7, 7, 7, 7, 7, 4, 0);

		__m256i ymm0_out0i, ymm1_out1i, ymm2_out2i, ymm3_out3i, ymm4_out4i, ymm5_out5i, ymm6_out6i, ymm7_out7i;

		ymm8_Krn = _mm256_broadcast_ss(fKernelV + 0);

		ymm0_out0 = _mm256_setzero_ps();
		ymm1_out1 = _mm256_setzero_ps();
		ymm2_out2 = _mm256_setzero_ps();
		ymm3_out3 = _mm256_setzero_ps();
		ymm4_out4 = _mm256_setzero_ps();
		ymm5_out5 = _mm256_setzero_ps();
		ymm6_out6 = _mm256_setzero_ps();
		ymm7_out7 = _mm256_setzero_ps();

		ymm0_out0 = _mm256_fmadd_ps(ymm8_Krn, _mm256_loadu_ps(pfSrc + 0), ymm0_out0);
		ymm1_out1 = _mm256_fmadd_ps(ymm8_Krn, _mm256_loadu_ps(pfSrc + (1 * iHShiftedStride)), ymm1_out1);
		ymm2_out2 = _mm256_fmadd_ps(ymm8_Krn, _mm256_loadu_ps(pfSrc + (2 * iHShiftedStride)), ymm2_out2);
		ymm3_out3 = _mm256_fmadd_ps(ymm8_Krn, _mm256_loadu_ps(pfSrc + (3 * iHShiftedStride)), ymm3_out3);
		ymm4_out4 = _mm256_fmadd_ps(ymm8_Krn, _mm256_loadu_ps(pfSrc + (4 * iHShiftedStride)), ymm4_out4);
		ymm5_out5 = _mm256_fmadd_ps(ymm8_Krn, _mm256_loadu_ps(pfSrc + (5 * iHShiftedStride)), ymm5_out5);
		ymm6_out6 = _mm256_fmadd_ps(ymm8_Krn, _mm256_loadu_ps(pfSrc + (6 * iHShiftedStride)), ymm6_out6);
		ymm7_out7 = _mm256_fmadd_ps(ymm8_Krn, _mm256_loadu_ps(pfSrc + (7 * iHShiftedStride)), ymm7_out7);

		for (int y = 1; y < 8; y++)
		{
			ymm8_Krn = _mm256_broadcast_ss(fKernelV + y);

			ymm0_out0 = _mm256_fmadd_ps(ymm8_Krn, _mm256_loadu_ps(pfSrc + (0 + y) * iHShiftedStride), ymm0_out0);
			ymm1_out1 = _mm256_fmadd_ps(ymm8_Krn, _mm256_loadu_ps(pfSrc + (1 + y) * iHShiftedStride), ymm1_out1);
			ymm2_out2 = _mm256_fmadd_ps(ymm8_Krn, _mm256_loadu_ps(pfSrc + (2 + y) * iHShiftedStride), ymm2_out2);
			ymm3_out3 = _mm256_fmadd_ps(ymm8_Krn, _mm256_loadu_ps(pfSrc + (3 + y) * iHShiftedStride), ymm3_out3);
			ymm4_out4 = _mm256_fmadd_ps(ymm8_Krn, _mm256_loadu_ps(pfSrc + (4 + y) * iHShiftedStride), ymm4_out4);
			ymm5_out5 = _mm256_fmadd_ps(ymm8_Krn, _mm256_loadu_ps(pfSrc + (5 + y) * iHShiftedStride), ymm5_out5);
			ymm6_out6 = _mm256_fmadd_ps(ymm8_Krn, _mm256_loadu_ps(pfSrc + (6 + y) * iHShiftedStride), ymm6_out6);
			ymm7_out7 = _mm256_fmadd_ps(ymm8_Krn, _mm256_loadu_ps(pfSrc + (7 + y) * iHShiftedStride), ymm7_out7);
		}

		ymm0_out0 = _mm256_max_ps(ymm0_out0, ymm9_MaxPS);
		ymm1_out1 = _mm256_max_ps(ymm1_out1, ymm9_MaxPS);
		ymm2_out2 = _mm256_max_ps(ymm2_out2, ymm9_MaxPS);
		ymm3_out3 = _mm256_max_ps(ymm3_out3, ymm9_MaxPS);
		ymm4_out4 = _mm256_max_ps(ymm4_out4, ymm9_MaxPS);
		ymm5_out5 = _mm256_max_ps(ymm5_out5, ymm9_MaxPS);
		ymm6_out6 = _mm256_max_ps(ymm6_out6, ymm9_MaxPS);
		ymm7_out7 = _mm256_max_ps(ymm7_out7, ymm9_MaxPS);

		ymm0_out0 = _mm256_min_ps(ymm0_out0, ymm10_MinPS);
		ymm1_out1 = _mm256_min_ps(ymm1_out1, ymm10_MinPS);
		ymm2_out2 = _mm256_min_ps(ymm2_out2, ymm10_MinPS);
		ymm3_out3 = _mm256_min_ps(ymm3_out3, ymm10_MinPS);
		ymm4_out4 = _mm256_min_ps(ymm4_out4, ymm10_MinPS);
		ymm5_out5 = _mm256_min_ps(ymm5_out5, ymm10_MinPS);
		ymm6_out6 = _mm256_min_ps(ymm6_out6, ymm10_MinPS);
		ymm7_out7 = _mm256_min_ps(ymm7_out7, ymm10_MinPS);

		ymm0_out0i = _mm256_cvtps_epi32(ymm0_out0);
		ymm1_out1i = _mm256_cvtps_epi32(ymm1_out1);
		ymm2_out2i = _mm256_cvtps_epi32(ymm2_out2);
		ymm3_out3i = _mm256_cvtps_epi32(ymm3_out3);
		ymm4_out4i = _mm256_cvtps_epi32(ymm4_out4);
		ymm5_out5i = _mm256_cvtps_epi32(ymm5_out5);
		ymm6_out6i = _mm256_cvtps_epi32(ymm6_out6);
		ymm7_out7i = _mm256_cvtps_epi32(ymm7_out7);

		ymm0_out0i = _mm256_packus_epi32(ymm0_out0i, ymm0_out0i);
		ymm1_out1i = _mm256_packus_epi32(ymm1_out1i, ymm1_out1i);
		ymm2_out2i = _mm256_packus_epi32(ymm2_out2i, ymm2_out2i);
		ymm3_out3i = _mm256_packus_epi32(ymm3_out3i, ymm3_out3i);
		ymm4_out4i = _mm256_packus_epi32(ymm4_out4i, ymm4_out4i);
		ymm5_out5i = _mm256_packus_epi32(ymm5_out5i, ymm5_out5i);
		ymm6_out6i = _mm256_packus_epi32(ymm6_out6i, ymm6_out6i);
		ymm7_out7i = _mm256_packus_epi32(ymm7_out7i, ymm7_out7i);

		ymm0_out0i = _mm256_packus_epi16(ymm0_out0i, ymm0_out0i);
		ymm1_out1i = _mm256_packus_epi16(ymm1_out1i, ymm1_out1i);
		ymm2_out2i = _mm256_packus_epi16(ymm2_out2i, ymm2_out2i);
		ymm3_out3i = _mm256_packus_epi16(ymm3_out3i, ymm3_out3i);
		ymm4_out4i = _mm256_packus_epi16(ymm4_out4i, ymm4_out4i);
		ymm5_out5i = _mm256_packus_epi16(ymm5_out5i, ymm5_out5i);
		ymm6_out6i = _mm256_packus_epi16(ymm6_out6i, ymm6_out6i);
		ymm7_out7i = _mm256_packus_epi16(ymm7_out7i, ymm7_out7i);

		ymm0_out0i = _mm256_permutevar8x32_epi32(ymm0_out0i, ymm11_8bit_perm);
		ymm1_out1i = _mm256_permutevar8x32_epi32(ymm1_out1i, ymm11_8bit_perm);
		ymm2_out2i = _mm256_permutevar8x32_epi32(ymm2_out2i, ymm11_8bit_perm);
		ymm3_out3i = _mm256_permutevar8x32_epi32(ymm3_out3i, ymm11_8bit_perm);
		ymm4_out4i = _mm256_permutevar8x32_epi32(ymm4_out4i, ymm11_8bit_perm);
		ymm5_out5i = _mm256_permutevar8x32_epi32(ymm5_out5i, ymm11_8bit_perm);
		ymm6_out6i = _mm256_permutevar8x32_epi32(ymm6_out6i, ymm11_8bit_perm);
		ymm7_out7i = _mm256_permutevar8x32_epi32(ymm7_out7i, ymm11_8bit_perm);
		
		_mm_storeu_si64(pucDst, _mm256_castsi256_si128(ymm0_out0i));
		_mm_storeu_si64(pucDst + 1 * iHVShiftedStride, _mm256_castsi256_si128(ymm1_out1i));
		_mm_storeu_si64(pucDst + 2 * iHVShiftedStride, _mm256_castsi256_si128(ymm2_out2i));
		_mm_storeu_si64(pucDst + 3 * iHVShiftedStride, _mm256_castsi256_si128(ymm3_out3i));
		_mm_storeu_si64(pucDst + 4 * iHVShiftedStride, _mm256_castsi256_si128(ymm4_out4i));
		_mm_storeu_si64(pucDst + 5 * iHVShiftedStride, _mm256_castsi256_si128(ymm5_out5i));
		_mm_storeu_si64(pucDst + 6 * iHVShiftedStride, _mm256_castsi256_si128(ymm6_out6i));
		_mm_storeu_si64(pucDst + 7 * iHVShiftedStride, _mm256_castsi256_si128(ymm7_out7i));

	}
	else // copy to out buf
	{
		float *pfSrc = CurrBlockShiftH_avx2 + 4 * iHShiftedStride;
		unsigned char *pucDst = CurrBlockShiftedHV_avx2;

		__m256 ymm0_out0, ymm1_out1, ymm2_out2, ymm3_out3, ymm4_out4, ymm5_out5, ymm6_out6, ymm7_out7;

		float fZero = 0.0f;
		__m256 ymm9_MaxPS = _mm256_broadcast_ss(&fZero);

		float f255 = 255.0f;
		__m256 ymm10_MinPS = _mm256_broadcast_ss(&f255);

		__m256i ymm11_8bit_perm = _mm256_set_epi32(7, 7, 7, 7, 7, 7, 4, 0);

		__m256i ymm0_out0i, ymm1_out1i, ymm2_out2i, ymm3_out3i, ymm4_out4i, ymm5_out5i, ymm6_out6i, ymm7_out7i;

		ymm0_out0 = _mm256_loadu_ps(pfSrc);
		ymm1_out1 = _mm256_loadu_ps(pfSrc + iHShiftedStride * 1);
		ymm2_out2 = _mm256_loadu_ps(pfSrc + iHShiftedStride * 2);
		ymm3_out3 = _mm256_loadu_ps(pfSrc + iHShiftedStride * 3);
		ymm4_out4 = _mm256_loadu_ps(pfSrc + iHShiftedStride * 4);
		ymm5_out5 = _mm256_loadu_ps(pfSrc + iHShiftedStride * 5);
		ymm6_out6 = _mm256_loadu_ps(pfSrc + iHShiftedStride * 6);
		ymm7_out7 = _mm256_loadu_ps(pfSrc + iHShiftedStride * 7);

		ymm0_out0 = _mm256_max_ps(ymm0_out0, ymm9_MaxPS);
		ymm1_out1 = _mm256_max_ps(ymm1_out1, ymm9_MaxPS);
		ymm2_out2 = _mm256_max_ps(ymm2_out2, ymm9_MaxPS);
		ymm3_out3 = _mm256_max_ps(ymm3_out3, ymm9_MaxPS);
		ymm4_out4 = _mm256_max_ps(ymm4_out4, ymm9_MaxPS);
		ymm5_out5 = _mm256_max_ps(ymm5_out5, ymm9_MaxPS);
		ymm6_out6 = _mm256_max_ps(ymm6_out6, ymm9_MaxPS);
		ymm7_out7 = _mm256_max_ps(ymm7_out7, ymm9_MaxPS);

		ymm0_out0 = _mm256_min_ps(ymm0_out0, ymm10_MinPS);
		ymm1_out1 = _mm256_min_ps(ymm1_out1, ymm10_MinPS);
		ymm2_out2 = _mm256_min_ps(ymm2_out2, ymm10_MinPS);
		ymm3_out3 = _mm256_min_ps(ymm3_out3, ymm10_MinPS);
		ymm4_out4 = _mm256_min_ps(ymm4_out4, ymm10_MinPS);
		ymm5_out5 = _mm256_min_ps(ymm5_out5, ymm10_MinPS);
		ymm6_out6 = _mm256_min_ps(ymm6_out6, ymm10_MinPS);
		ymm7_out7 = _mm256_min_ps(ymm7_out7, ymm10_MinPS);

		ymm0_out0i = _mm256_cvtps_epi32(ymm0_out0);
		ymm1_out1i = _mm256_cvtps_epi32(ymm1_out1);
		ymm2_out2i = _mm256_cvtps_epi32(ymm2_out2);
		ymm3_out3i = _mm256_cvtps_epi32(ymm3_out3);
		ymm4_out4i = _mm256_cvtps_epi32(ymm4_out4);
		ymm5_out5i = _mm256_cvtps_epi32(ymm5_out5);
		ymm6_out6i = _mm256_cvtps_epi32(ymm6_out6);
		ymm7_out7i = _mm256_cvtps_epi32(ymm7_out7);

		ymm0_out0i = _mm256_packus_epi32(ymm0_out0i, ymm0_out0i);
		ymm1_out1i = _mm256_packus_epi32(ymm1_out1i, ymm1_out1i);
		ymm2_out2i = _mm256_packus_epi32(ymm2_out2i, ymm2_out2i);
		ymm3_out3i = _mm256_packus_epi32(ymm3_out3i, ymm3_out3i);
		ymm4_out4i = _mm256_packus_epi32(ymm4_out4i, ymm4_out4i);
		ymm5_out5i = _mm256_packus_epi32(ymm5_out5i, ymm5_out5i);
		ymm6_out6i = _mm256_packus_epi32(ymm6_out6i, ymm6_out6i);
		ymm7_out7i = _mm256_packus_epi32(ymm7_out7i, ymm7_out7i);

		ymm0_out0i = _mm256_packus_epi16(ymm0_out0i, ymm0_out0i);
		ymm1_out1i = _mm256_packus_epi16(ymm1_out1i, ymm1_out1i);
		ymm2_out2i = _mm256_packus_epi16(ymm2_out2i, ymm2_out2i);
		ymm3_out3i = _mm256_packus_epi16(ymm3_out3i, ymm3_out3i);
		ymm4_out4i = _mm256_packus_epi16(ymm4_out4i, ymm4_out4i);
		ymm5_out5i = _mm256_packus_epi16(ymm5_out5i, ymm5_out5i);
		ymm6_out6i = _mm256_packus_epi16(ymm6_out6i, ymm6_out6i);
		ymm7_out7i = _mm256_packus_epi16(ymm7_out7i, ymm7_out7i);

		ymm0_out0i = _mm256_permutevar8x32_epi32(ymm0_out0i, ymm11_8bit_perm);
		ymm1_out1i = _mm256_permutevar8x32_epi32(ymm1_out1i, ymm11_8bit_perm);
		ymm2_out2i = _mm256_permutevar8x32_epi32(ymm2_out2i, ymm11_8bit_perm);
		ymm3_out3i = _mm256_permutevar8x32_epi32(ymm3_out3i, ymm11_8bit_perm);
		ymm4_out4i = _mm256_permutevar8x32_epi32(ymm4_out4i, ymm11_8bit_perm);
		ymm5_out5i = _mm256_permutevar8x32_epi32(ymm5_out5i, ymm11_8bit_perm);
		ymm6_out6i = _mm256_permutevar8x32_epi32(ymm6_out6i, ymm11_8bit_perm);
		ymm7_out7i = _mm256_permutevar8x32_epi32(ymm7_out7i, ymm11_8bit_perm);

		_mm_storeu_si64(pucDst, _mm256_castsi256_si128(ymm0_out0i));
		_mm_storeu_si64(pucDst + 1 * iHVShiftedStride, _mm256_castsi256_si128(ymm1_out1i));
		_mm_storeu_si64(pucDst + 2 * iHVShiftedStride, _mm256_castsi256_si128(ymm2_out2i));
		_mm_storeu_si64(pucDst + 3 * iHVShiftedStride, _mm256_castsi256_si128(ymm3_out3i));
		_mm_storeu_si64(pucDst + 4 * iHVShiftedStride, _mm256_castsi256_si128(ymm4_out4i));
		_mm_storeu_si64(pucDst + 5 * iHVShiftedStride, _mm256_castsi256_si128(ymm5_out5i));
		_mm_storeu_si64(pucDst + 6 * iHVShiftedStride, _mm256_castsi256_si128(ymm6_out6i));
		_mm_storeu_si64(pucDst + 7 * iHVShiftedStride, _mm256_castsi256_si128(ymm7_out7i));
	}
}


void SubShiftBlock4x4_KS8_avx2(float *fKernelH, float *fKernelV)
{
	const int iSrcStride = 12; // 4+4+4 - 4 margins + 4 block size
	const int iHShiftedStride = 4;
	const int iHVShiftedStride = 4;

	if (fKernelH != 0)
	{
		for (int j = 0; j < (iBlockSizeY + iKS); j++)
		{
			for (int i = 0; i < iBlockSizeX; i++)
			{
				float fOut = 0.0f;

				for (int k = 0; k < iKS; k++)
				{
					float fSample = (float)CurrBlock[j * nSrcPitch[0] + i + k];
					fOut += fSample * fKernelH[k];
				}

				CurrBlockShiftH[j * iBlockSizeX + i] = fOut;
			}
		}


		unsigned char *pSrc = CurrBlock;
		float *pDst = CurrBlockShiftH_avx2;
		__m256 ymm_Krn;
		__m256i ymm_perm_rot_1_float_left = _mm256_set_epi32(0, 7, 6, 5, 4, 3, 2, 1);

		__m256 ymm0_row0_0f, ymm1_row0_1f;
		__m256 ymm2_row1_0f, ymm3_row1_1f;
		__m256 ymm4_row2_0f, ymm5_row2_1f;
		__m256 ymm6_row3_0f, ymm7_row3_1f;

		for (int y = 0; y < 3; y++) // 3 groups of 4 rows
		{
			__m256 ymm8_out_row0 = _mm256_setzero_ps();
			__m256 ymm9_out_row1 = _mm256_setzero_ps();
			__m256 ymm10_out_row2 = _mm256_setzero_ps();
			__m256 ymm11_out_row3 = _mm256_setzero_ps();

			ymm0_row0_0f = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadu_si64(pSrc)));
			ymm1_row0_1f = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadu_si64(pSrc + 8)));

			ymm2_row1_0f = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadu_si64(pSrc + iSrcStride)));
			ymm3_row1_1f = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadu_si64(pSrc + iSrcStride + 8)));

			ymm4_row2_0f = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadu_si64(pSrc + iSrcStride * 2)));
			ymm5_row2_1f = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadu_si64(pSrc + iSrcStride * 2 + 8)));

			ymm6_row3_0f = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadu_si64(pSrc + iSrcStride * 3)));
			ymm7_row3_1f = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadu_si64(pSrc + iSrcStride * 3 + 8)));

			ymm_Krn = _mm256_broadcast_ss(fKernelH);

			// 1 of 8
			ymm8_out_row0 = _mm256_fmadd_ps(ymm0_row0_0f, ymm_Krn, ymm8_out_row0);
			ymm9_out_row1 = _mm256_fmadd_ps(ymm2_row1_0f, ymm_Krn, ymm9_out_row1);
			ymm10_out_row2 = _mm256_fmadd_ps(ymm4_row2_0f, ymm_Krn, ymm10_out_row2);
			ymm11_out_row3 = _mm256_fmadd_ps(ymm6_row3_0f, ymm_Krn, ymm11_out_row3);

			for (int x = 1; x < 4; x++)
			{
				//shift 1 float sample
				ymm0_row0_0f = _mm256_permutevar8x32_ps(ymm0_row0_0f, ymm_perm_rot_1_float_left);
				ymm1_row0_1f = _mm256_permutevar8x32_ps(ymm1_row0_1f, ymm_perm_rot_1_float_left);
				ymm0_row0_0f = _mm256_blend_ps(ymm0_row0_0f, ymm1_row0_1f, 128);

				ymm2_row1_0f = _mm256_permutevar8x32_ps(ymm2_row1_0f, ymm_perm_rot_1_float_left);
				ymm3_row1_1f = _mm256_permutevar8x32_ps(ymm3_row1_1f, ymm_perm_rot_1_float_left);
				ymm2_row1_0f = _mm256_blend_ps(ymm2_row1_0f, ymm3_row1_1f, 128);

				ymm4_row2_0f = _mm256_permutevar8x32_ps(ymm4_row2_0f, ymm_perm_rot_1_float_left);
				ymm5_row2_1f = _mm256_permutevar8x32_ps(ymm5_row2_1f, ymm_perm_rot_1_float_left);
				ymm4_row2_0f = _mm256_blend_ps(ymm4_row2_0f, ymm5_row2_1f, 128);

				ymm6_row3_0f = _mm256_permutevar8x32_ps(ymm6_row3_0f, ymm_perm_rot_1_float_left);
				ymm7_row3_1f = _mm256_permutevar8x32_ps(ymm7_row3_1f, ymm_perm_rot_1_float_left);
				ymm6_row3_0f = _mm256_blend_ps(ymm6_row3_0f, ymm7_row3_1f, 128);

				ymm_Krn = _mm256_broadcast_ss(fKernelH + x);

				ymm8_out_row0 = _mm256_fmadd_ps(ymm0_row0_0f, ymm_Krn, ymm8_out_row0);
				ymm9_out_row1 = _mm256_fmadd_ps(ymm2_row1_0f, ymm_Krn, ymm9_out_row1);
				ymm10_out_row2 = _mm256_fmadd_ps(ymm4_row2_0f, ymm_Krn, ymm10_out_row2);
				ymm11_out_row3 = _mm256_fmadd_ps(ymm6_row3_0f, ymm_Krn, ymm11_out_row3);

			}

			// store need to be truncated to 12 floats !
/*			_mm256_storeu_ps(pDst, ymm8_out_row0);
			_mm256_storeu_ps(pDst + 8, ymm9_out_row1);
			_mm256_storeu_ps(pDst + 16, ymm10_out_row2);
			_mm256_storeu_ps(pDst + 24, ymm11_out_row3);*/

			pSrc = pSrc + (iSrcStride * 4); // in bytes
			pDst = pDst + (iHShiftedStride * 4); // in float32

		}
	}
	else // copy to CurrBlockShiftH temp buf
	{
		unsigned char *pSrc = CurrBlock + 4; // iKS_d2
		float *pDst = CurrBlockShiftH_avx2;

		// block is 4x4, + 4*2 margins = 12 lines to convert-copy
		_mm_storeu_ps(pDst, _mm_cvtepi32_ps(_mm_cvtepu8_epi32(_mm_loadu_si32(pSrc + 0))));
		_mm_storeu_ps(pDst + iHShiftedStride * 1, _mm_cvtepi32_ps(_mm_cvtepu8_epi32(_mm_loadu_si32(pSrc + iSrcStride * 1))));
		_mm_storeu_ps(pDst + iHShiftedStride * 2, _mm_cvtepi32_ps(_mm_cvtepu8_epi32(_mm_loadu_si32(pSrc + iSrcStride * 2))));
		_mm_storeu_ps(pDst + iHShiftedStride * 3, _mm_cvtepi32_ps(_mm_cvtepu8_epi32(_mm_loadu_si32(pSrc + iSrcStride * 3))));
		_mm_storeu_ps(pDst + iHShiftedStride * 4, _mm_cvtepi32_ps(_mm_cvtepu8_epi32(_mm_loadu_si32(pSrc + iSrcStride * 4))));
		_mm_storeu_ps(pDst + iHShiftedStride * 5, _mm_cvtepi32_ps(_mm_cvtepu8_epi32(_mm_loadu_si32(pSrc + iSrcStride * 5))));
		_mm_storeu_ps(pDst + iHShiftedStride * 6, _mm_cvtepi32_ps(_mm_cvtepu8_epi32(_mm_loadu_si32(pSrc + iSrcStride * 6))));
		_mm_storeu_ps(pDst + iHShiftedStride * 7, _mm_cvtepi32_ps(_mm_cvtepu8_epi32(_mm_loadu_si32(pSrc + iSrcStride * 7))));
		_mm_storeu_ps(pDst + iHShiftedStride * 8, _mm_cvtepi32_ps(_mm_cvtepu8_epi32(_mm_loadu_si32(pSrc + iSrcStride * 8))));
		_mm_storeu_ps(pDst + iHShiftedStride * 9, _mm_cvtepi32_ps(_mm_cvtepu8_epi32(_mm_loadu_si32(pSrc + iSrcStride * 9))));
		_mm_storeu_ps(pDst + iHShiftedStride * 10, _mm_cvtepi32_ps(_mm_cvtepu8_epi32(_mm_loadu_si32(pSrc + iSrcStride * 10))));
		_mm_storeu_ps(pDst + iHShiftedStride * 11, _mm_cvtepi32_ps(_mm_cvtepu8_epi32(_mm_loadu_si32(pSrc + iSrcStride * 11))));
		
	}

	if (fKernelV != 0)
	{
		// V shift
		for (int i = 0; i < iBlockSizeX; i++)
		{
			for (int j = 0; j < iBlockSizeY; j++)
			{
				float fOut = 0.0f;

				for (int k = 0; k < iKS; k++)
				{
					float fSample = CurrBlockShiftH[(j + k) * iBlockSizeX + i];
					fOut += fSample * fKernelV[k];
				}

				fOut += 0.5f;

				if (fOut > 255.0f) fOut = 255.0f;
				if (fOut < 0.0f) fOut = 0.0f;

				CurrBlockShiftedHV[j * iBlockSizeX + i] = (unsigned char)(fOut);
			}
		}

		// V shift - AVX2 can process 2 4x4 blocks at once ?? need load 4 + 4 columns from different planes ? need separate shift function 4x4_UV
		float *pfSrc = CurrBlockShiftH_avx2;
		unsigned char *pucDst = CurrBlockShiftedHV_avx2;

		__m128 xmm0_out0, xmm1_out1, xmm2_out2, xmm3_out3;
		__m128 xmm8_Krn;

		float fZero = 0.0f;
		__m128 xmm9_MaxPS = _mm_broadcast_ss(&fZero);

		float f255 = 255.0f;
		__m128 xmm10_MinPS = _mm_broadcast_ss(&f255);

//		__m128i ymm11_8bit_perm = _mm256_set_epi32(7, 7, 7, 7, 7, 7, 4, 0);

		__m128i xmm0_out0i, xmm1_out1i, xmm2_out2i, xmm3_out3i;

		xmm8_Krn = _mm_broadcast_ss(fKernelV + 0);

		xmm0_out0 = _mm_setzero_ps();
		xmm1_out1 = _mm_setzero_ps();
		xmm2_out2 = _mm_setzero_ps();
		xmm3_out3 = _mm_setzero_ps();

		xmm0_out0 = _mm_fmadd_ps(xmm8_Krn, _mm_loadu_ps(pfSrc + 0), xmm0_out0);
		xmm1_out1 = _mm_fmadd_ps(xmm8_Krn, _mm_loadu_ps(pfSrc + (1 * iHShiftedStride)), xmm1_out1);
		xmm2_out2 = _mm_fmadd_ps(xmm8_Krn, _mm_loadu_ps(pfSrc + (2 * iHShiftedStride)), xmm2_out2);
		xmm3_out3 = _mm_fmadd_ps(xmm8_Krn, _mm_loadu_ps(pfSrc + (3 * iHShiftedStride)), xmm3_out3);

		for (int y = 1; y < 4; y++)
		{
			xmm8_Krn = _mm_broadcast_ss(fKernelV + y);

			xmm0_out0 = _mm_fmadd_ps(xmm8_Krn, _mm_loadu_ps(pfSrc + (0 + y) * iHShiftedStride), xmm0_out0);
			xmm1_out1 = _mm_fmadd_ps(xmm8_Krn, _mm_loadu_ps(pfSrc + (1 + y) * iHShiftedStride), xmm1_out1);
			xmm2_out2 = _mm_fmadd_ps(xmm8_Krn, _mm_loadu_ps(pfSrc + (2 + y) * iHShiftedStride), xmm2_out2);
			xmm3_out3 = _mm_fmadd_ps(xmm8_Krn, _mm_loadu_ps(pfSrc + (3 + y) * iHShiftedStride), xmm3_out3);
		}

		xmm0_out0 = _mm_max_ps(xmm0_out0, xmm9_MaxPS);
		xmm1_out1 = _mm_max_ps(xmm1_out1, xmm9_MaxPS);
		xmm2_out2 = _mm_max_ps(xmm2_out2, xmm9_MaxPS);
		xmm3_out3 = _mm_max_ps(xmm3_out3, xmm9_MaxPS);

		xmm0_out0 = _mm_min_ps(xmm0_out0, xmm10_MinPS);
		xmm1_out1 = _mm_min_ps(xmm1_out1, xmm10_MinPS);
		xmm2_out2 = _mm_min_ps(xmm2_out2, xmm10_MinPS);
		xmm3_out3 = _mm_min_ps(xmm3_out3, xmm10_MinPS);

		xmm0_out0i = _mm_cvtps_epi32(xmm0_out0);
		xmm1_out1i = _mm_cvtps_epi32(xmm1_out1);
		xmm2_out2i = _mm_cvtps_epi32(xmm2_out2);
		xmm3_out3i = _mm_cvtps_epi32(xmm3_out3);

		xmm0_out0i = _mm_packus_epi32(xmm0_out0i, xmm0_out0i);
		xmm1_out1i = _mm_packus_epi32(xmm1_out1i, xmm1_out1i);
		xmm2_out2i = _mm_packus_epi32(xmm2_out2i, xmm2_out2i);
		xmm3_out3i = _mm_packus_epi32(xmm3_out3i, xmm3_out3i);

		xmm0_out0i = _mm_packus_epi16(xmm0_out0i, xmm0_out0i);
		xmm1_out1i = _mm_packus_epi16(xmm1_out1i, xmm1_out1i);
		xmm2_out2i = _mm_packus_epi16(xmm2_out2i, xmm2_out2i);
		xmm3_out3i = _mm_packus_epi16(xmm3_out3i, xmm3_out3i);

		_mm_storeu_si32(pucDst, xmm0_out0i);
		_mm_storeu_si32(pucDst + 1 * iHVShiftedStride, xmm1_out1i);
		_mm_storeu_si32(pucDst + 2 * iHVShiftedStride, xmm2_out2i);
		_mm_storeu_si32(pucDst + 3 * iHVShiftedStride, xmm3_out3i);

	}
	else // copy to out buf
	{
		float *pfSrc = CurrBlockShiftH_avx2 + 4 * iHShiftedStride;
		unsigned char *pucDst = CurrBlockShiftedHV_avx2;

		__m128 xmm0_out0, xmm1_out1, xmm2_out2, xmm3_out3;

		float fZero = 0.0f;
		__m128 xmm9_MaxPS = _mm_broadcast_ss(&fZero);

		float f255 = 255.0f;
		__m128 xmm10_MinPS = _mm_broadcast_ss(&f255);

		__m128i xmm0_out0i, xmm1_out1i, xmm2_out2i, xmm3_out3i;

		xmm0_out0 = _mm_loadu_ps(pfSrc);
		xmm1_out1 = _mm_loadu_ps(pfSrc + iHShiftedStride * 1);
		xmm2_out2 = _mm_loadu_ps(pfSrc + iHShiftedStride * 2);
		xmm3_out3 = _mm_loadu_ps(pfSrc + iHShiftedStride * 3);

		xmm0_out0 = _mm_max_ps(xmm0_out0, xmm9_MaxPS);
		xmm1_out1 = _mm_max_ps(xmm1_out1, xmm9_MaxPS);
		xmm2_out2 = _mm_max_ps(xmm2_out2, xmm9_MaxPS);
		xmm3_out3 = _mm_max_ps(xmm3_out3, xmm9_MaxPS);

		xmm0_out0 = _mm_min_ps(xmm0_out0, xmm10_MinPS);
		xmm1_out1 = _mm_min_ps(xmm1_out1, xmm10_MinPS);
		xmm2_out2 = _mm_min_ps(xmm2_out2, xmm10_MinPS);
		xmm3_out3 = _mm_min_ps(xmm3_out3, xmm10_MinPS);

		xmm0_out0i = _mm_cvtps_epi32(xmm0_out0);
		xmm1_out1i = _mm_cvtps_epi32(xmm1_out1);
		xmm2_out2i = _mm_cvtps_epi32(xmm2_out2);
		xmm3_out3i = _mm_cvtps_epi32(xmm3_out3);

		xmm0_out0i = _mm_packus_epi32(xmm0_out0i, xmm0_out0i);
		xmm1_out1i = _mm_packus_epi32(xmm1_out1i, xmm1_out1i);
		xmm2_out2i = _mm_packus_epi32(xmm2_out2i, xmm2_out2i);
		xmm3_out3i = _mm_packus_epi32(xmm3_out3i, xmm3_out3i);

		xmm0_out0i = _mm_packus_epi16(xmm0_out0i, xmm0_out0i);
		xmm1_out1i = _mm_packus_epi16(xmm1_out1i, xmm1_out1i);
		xmm2_out2i = _mm_packus_epi16(xmm2_out2i, xmm2_out2i);
		xmm3_out3i = _mm_packus_epi16(xmm3_out3i, xmm3_out3i);

		_mm_storeu_si32(pucDst, xmm0_out0i);
		_mm_storeu_si32(pucDst + 1 * iHVShiftedStride, xmm1_out1i);
		_mm_storeu_si32(pucDst + 2 * iHVShiftedStride, xmm2_out2i);
		_mm_storeu_si32(pucDst + 3 * iHVShiftedStride, xmm3_out3i);
	}
}



int main()
{
	for (int j = 0; j < 16; j++)
	{
		for (int i = 0; i < 16; i++)
		{
//			CurrBlock[j * 16 + i] = i + j * 10;
			CurrBlock[j * nSrcPitch[0] + i] = (unsigned char)((0.25f * (sinf(i * fFreqH * fPi) + 0.25f * sinf(j * fFreqV * fPi)) + 0.5f) * 255.0f);
		}
	}

	unsigned char *pucRef = &RefPlane[0]; // upper left corner
	unsigned char *pucCurr = &CurrBlock[0];

	CalcShiftKernel(fKernelH_01, 0.25f, iKS);
	CalcShiftKernel(fKernelH_10, 0.5f, iKS);
	CalcShiftKernel(fKernelH_11, 0.75f, iKS);
	
 	CalcShiftKernel(fKernelV_01, 0.25f, iKS);
	CalcShiftKernel(fKernelV_10, 0.5f, iKS);
	CalcShiftKernel(fKernelV_11, 0.75f, iKS);

	float *pfKrnH = 0;
	float *pfKrnV = 0;

	switch (iPelH)
	{
	case 0:
		pfKrnH = 0;
		break;
	case 1:
		pfKrnH = fKernelH_01;
		break;
	case 2:
		pfKrnH = fKernelH_10;
		break;
	case 3:
		pfKrnH = fKernelH_11;
		break;
	}

	switch (iPelV)
	{
	case 0:
		pfKrnV = 0;
		break;
	case 1:
		pfKrnV = fKernelV_01;
		break;
	case 2:
		pfKrnV = fKernelV_10;
		break;
	case 3:
		pfKrnV = fKernelV_11;
		break;
	}

//	SubShiftBlock(pfKrnH, pfKrnV);
	SubShiftBlock8x8_KS8_avx2(pfKrnH, pfKrnV);

	return 0;
}

