This version has added:
- transforms are done via homogeneous matrix operations (allows us to collapse our many different operations into a couple of matrix multiplications)
- clipping has been added for out-of-bounds objects and vertices. objects that are partially in bounds will have their vertices clipped appropriately (and will have additional vertices generated as needed when specific vertices lie out of bounds)
- the FOV/clipping volume is currently hard-coded: meaning that there is a forced FOV of 90 degrees, and the camera currently views slightly out of the bounds so that clipping can be seen a little easier (and for when i was debugging the clipping of vertices)

 <video loop src="chapter-7-demo.mp4"> demo video </video> 
