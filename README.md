# Color-Rate-Calculator
Python application which calculates the rate of dark and light parts of objects in the input picture.

For accurate results, the input image shouldn't have a very colourfull background.
The bigger part of the image the object, the better the results are.
To avoid segmentation, the input image should have a one colour background.

Packages used: 

 - opencv
 - tkinter
 - math
 - numpy


How to use:

 0.  [optional step] Before launch, cut the important part out from the original picture to make the calculation more accurate. 
 1.  Launch program, choose image.
 2.  3 window appears: a hsv colour palette, the original input image, a window which will show the segmented area.
 3.  Let's designate a black part of the object. Can be done by one click, or by designating with a rectangle by holding the left mouse button.
 4.  We can make the segmented area more accurate by h, s, v,  Shift + h, Shift + s, Shift + v buttons. The hsv colour palette may help for experts, but it's easy to reach optimal result without understanding it.
 5.  Once the segmentated area is accurate enough, we can save it in a variable. In case of black part: press b button. 
 6.  Designating the white part works the same way. Once it is ready press w button to save. (The colour sequence is optional, and works both way)
 7.  If the 2 aera is ready and the b and w button was pushed, the results appear in the console. 
