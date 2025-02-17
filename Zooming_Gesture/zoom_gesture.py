import cv2
from cvzone.HandTrackingModule import HandDetector

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # Width
cap.set(4, 720)   # Height

# Initialize hand detector
detector = HandDetector(detectionCon=0.8)

# Variables for zooming
startDist = None
scaleFactor = 1  # Start with 1 (original size)
cx, cy = 640, 360  # Center position

# Load the image
img1 = cv2.imread('doremon.jpg')
if img1 is None:
    print("Error: Image not found!")
    exit()

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)

    if len(hands) == 2:
        # Check for zoom gesture (index & middle fingers up)
        if detector.fingersUp(hands[0]) == [1, 1, 0, 0, 0] and detector.fingersUp(hands[1]) == [1, 1, 0, 0, 0]:
            length, info, img = detector.findDistance(hands[0]["center"], hands[1]["center"], img)
            
            if startDist is None:
                startDist = length  # Set initial reference distance

            # Dynamic scale adjustment
            scaleFactor = max(0.5, min(3, length / startDist))  # Ensures scale is between 0.5x to 3x
            cx, cy = info[4:]  # Update center position

            print(f"Scale Factor: {scaleFactor:.2f}, Distance: {length:.2f}")

    else:
        startDist = None  # Reset when hands are not detected

    try:
        # Resize image based on scaleFactor
        h1, w1, _ = img1.shape
        newH = int(h1 * scaleFactor)
        newW = int(w1 * scaleFactor)
        img_resized = cv2.resize(img1, (newW, newH))

        # Ensure image stays inside the frame
        y1, y2 = max(0, cy-newH//2), min(img.shape[0], cy+newH//2)
        x1, x2 = max(0, cx-newW//2), min(img.shape[1], cx+newW//2)

        # Crop overlay to fit in the camera frame
        img[y1:y2, x1:x2] = img_resized[:y2-y1, :x2-x1]

    except Exception as e:
        print("Error:", e)

    cv2.imshow("Image", img)
    
    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
