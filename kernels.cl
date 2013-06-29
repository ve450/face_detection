typedef struct testT {
  int left;
  int right;
  int res;
} testT;

__kernel void SAXPY(__global testT *a, uint tmp) {
  const int i = get_global_id(0);
  a[i].res = a[i].left + a[i].right + tmp;
}
