import cv2

# Load the image
image = cv2.imread('clean_light.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                               cv2.THRESH_BINARY_INV, 11, 4)

# Find contours
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

rectangles = []
max_area = 0
biggest_rect = None

for cnt in contours:
    approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
    if len(approx) == 4 and cv2.contourArea(cnt) > 100:  # it's a rectangle
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        rectangles.append((x, y, w, h))
        if area > max_area:
            max_area = area
            biggest_rect = (x, y, w, h)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

print("Rectangles found:", len(rectangles))

# Crop the biggest rectangle
if biggest_rect is not None:
    x, y, w, h = biggest_rect
    cropped = image[y:y+h, x:x+w]
    cv2.imshow("Cropped Biggest Rectangle", cropped)
    cv2.imwrite("cropped_biggest_rectangle.png", cropped)  # Save the cropped image

# Show the result
cv2.imshow("Rectangles", image)
cv2.waitKey(0)
cv2.destroyAllWindows()