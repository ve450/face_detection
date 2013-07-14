#include <stdarg.h>
#include <string>
#include <string.h>
#include <vector>
using namespace std;
//combine multiple cv::Mat together, into an array of uchar
//usage: 
//	EXAMPLE:
//	 uchar* buff; int total; string header;
//       buff = encodeMatirx ( total, header, Mat_1, Mat_2, Mat_3, Mat_4)
//       The header format will be : "col#1-row#1-type1/col#2-row#2-type2/col#2-row#2-type2/col#2-row#2-type2/"
//       Type is a enum defined in opencv, the way to interpret the integer into specific type can be found at this URL
//       http://stackoverflow.com/questions/12335663/getting-enum-names-e-g-cv-32fc1-of-opencv-image-types
// NOTE: users must remeber to release the memory of buff to avoid memory leak!


uchar* encodeMatrix (int &size, int* &header, int n, ...) 
{
	va_list Mats;
	Mat** MatsPtr = new Mat*[n];
	va_start ( Mats, n);
	int totalSize = 0;
	header = new int[1 + n * 3];
	header[0] = n;
	for ( int i = 0; i < n; i++)
	{
		MatsPtr[i] = va_arg ( Mats, cv::Mat*);
		totalSize += MatsPtr[i]->cols * MatsPtr[i]->rows * MatsPtr[i]->elemSize(); 
		header[1 + i * 3] = MatsPtr[i]->cols; //record size in col
		header[1 + i * 3 + 1] = MatsPtr[i]->rows;//record size in row
		header[1 + i * 3 + 2] = MatsPtr[i]->type();//record elem_type
	}
	size = totalSize;
	uchar* result = new uchar[totalSize];
	memset ( result , 0, totalSize);
	int offset = 0;
	for ( int i = 0; i < n; i++)
 	{
		memcpy ( result + offset, MatsPtr[i]->data, MatsPtr[i]->cols * MatsPtr[i]->rows * MatsPtr[i]->elemSize());
		offset += MatsPtr[i]->cols * MatsPtr[i]->rows * MatsPtr[i]->elemSize();
	}
	delete [] MatsPtr;
	return result;
}

uchar* encodeMatrix (int &size, int* &header, vector<Mat*> mats)
{
	int totalSize = 0;
	header = new int[1 + mats.size() * 3];
	header[0] = mats.size();
	for ( int i = 0; i < mats.size(); i++)
	{//printf("mat %d, %d  *   %d \n", i+1, mats[i]->cols, mats[i]->rows);
		totalSize += mats[i]->cols * mats[i]->rows * mats[i]->elemSize(); 
		header[1 + i * 3] = mats[i]->cols; //record size in col
		header[1 + i * 3 + 1] = mats[i]->rows;//record size in row
		header[1 + i * 3 + 2] = mats[i]->type();//record elem_type
	}
	size = totalSize;
	uchar* result = new uchar[totalSize];
	memset ( result , 0, totalSize);
	int offset = 0;
	for ( int i = 0; i < mats.size(); i++)
 	{//printf("address is %ld\n", (unsigned long int)mats[i]->data);
		memcpy ( result + offset, mats[i]->data, mats[i]->cols * mats[i]->rows * mats[i]->elemSize());
		offset += mats[i]->cols * mats[i]->rows * mats[i]->elemSize();
	}
	return result;
}
int** decodeMatrixForInt(int* header, uchar *data)
{
	int n = header[0];
	int** result = new int*[n];
	int count = 0;
	for( int i = 0; i < n; i++) 
	{
		int col = header[1 + i * 3];
		int row = header[1 + i * 3 + 1];
		assert(header[1 + i * 3 + 2] == 4);
		result[i] =(int*)(data + count);
		count += col * row * 4;
	}
	return result;
}		
double** decodeMatrixForDouble(int* header, uchar *data)
{
	int n = header[0];
	double** result = new double*[n];
	int count = 0;
	for( int i = 0; i < n; i++) 
	{
		int col = header[1 + i * 3];
		int row = header[1 + i * 3 + 1];
		assert(header[1 + i * 3 + 2] == 6);
		result[i] = (double*)(data + count);
		count += col * row * 8;
	}
	return result;
}		
/*
void decodeMatrix (int &size, string &header, int n, uchar *content) 
{
	//va_list Mats;
	//Mat** MatsPtr = new Mat*[n];
	//va_start ( Mats, n);
	//int totalSize = 0;
  int i=0;
  int cols[n], rows[n], type[n];
  std::size_t found = header.find_first_of("/");
  while (found!=std::string::npos)
  {
    //int cols, rows, type;
		sscanf (header.substr(0,found).c_str(), "%d-%d-%d", 
            &(cols[i]), &(rows[i]), &(type[i]));
    i++;
    if (i>=n) break;
    //printf("%s\n", header.substr(0, found).c_str());
    //printf("%d,%d,%d\n", cols,rows,type);
    header.erase(0, found+1);
    found=header.find_first_of("/");
  }
 
	int offset = 0;
	for ( int i = 0; i < n; i++)
	{
    int elemSize = //based on type[i]
		int matSize= cols[i] * rows[i] * elemSize; 
    uchar buf[matSize];
	  memset ( buf , 0, matSize);
		memcpy ( buf, content + offset, matSize);
		offset += matSize;
	}
  
}*/


