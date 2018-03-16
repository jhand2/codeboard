# Code Board
A mobile application that can take a picture of code on a white board and run it in real time

Machine learning aspects will run on a Flask server using tensorflow. Actual learning will likely be done offline.

## MVP Features
* Only allow a single function
* C, C++, Java, maybe python
* Allow input arguments

## Dataset
https://www.kaggle.com/xainano/handwrittenmathsymbols


### Missing symbols
* times (*)
* mod (%)
* period (.)
* semi-colon (;)

### How I created each missing symbol artificially

See `expand_data.py` for the code I used to do this. Its a simple command line tool, run `python3 expand_data.py` to see usage

#### times
Take X, rotate 45 degrees, bitwise & the original with the result

#### mod
Rotate divide 45 degrees

#### period
White out the top 80-90% of the exclamation point

#### semi-colon
Flip exclamation point and distort the bottom 50% to progressively curve left

