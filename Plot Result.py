import cv2

x = cv2.imread(".\\DataBase\\06100.jpg")
cv2.circle(x,(615,289),3,(0,0,255), -1)
cv2.circle(x,(654,289),3,(0,0,255), -1)
cv2.circle(x,(640,245),3,(0,0,245), -1)
cv2.circle(x,(629,245),3,(0,0,255), -1)
cv2.imshow("Window", x)

cv2.waitKey(0)
cv2.destroyAllWindows()