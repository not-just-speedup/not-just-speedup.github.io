/* PolyBench/C 4.2.1 reference kernel, unmodified. Copyright (c) 2010-2016 Ohio State University.
   Distributed under the Ohio State University Software Distribution License: see POLYBENCH-LICENSE.txt. */

static
void kernel_floyd_warshall(int n,
			   DATA_TYPE POLYBENCH_2D(path,N,N,n,n))
{
  int i, j, k;

#pragma scop
  for (k = 0; k < _PB_N; k++)
    {
      for(i = 0; i < _PB_N; i++)
	for (j = 0; j < _PB_N; j++)
	  path[i][j] = path[i][j] < path[i][k] + path[k][j] ?
	    path[i][j] : path[i][k] + path[k][j];
    }
#pragma endscop

}