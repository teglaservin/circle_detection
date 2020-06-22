# Circle Detection from contour points using GPU accelerated RANSAC algorithm.

The following project takes a list of coordinates as input and returns the best fitting circle for those coordinates, generated from a GPU accelerated RANSAC algorithm.

## Installation

Install the latest Cuda and the corresponding Visual Studio. For ease of execution set a custom command prompt in Visual Studio.

## Execution

First build the project via:

```
nvcc -ccbin "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.20.27508\bin\Hostx86\x64" circle.cu -o circle
```

Be sure to replace any directory details if your Visual Studio is installed in a different directory or is running on a different version.

Then execute via:

```
circle.exe "path/to/txt"
```

Where the txt file referenced in the command line parameter includes the coorindates of the input contour points, as follows (each line has a set of coordinates separated by a space character):

```
245 435
246 345
...
```
