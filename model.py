from ultralytics import YOLO
import cv2

# Load YOLOv8 model
model = YOLO('best.pt')  # replace with your path if needed

# Open the webcam
cap = cv2.VideoCapture(0)  # 0 for default webcam, change if using external cam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform detection
    results = model(frame)

    # Draw results on the frame
    annotated_frame = results[0].plot()  # uses built-in plotting with boxes and labels

    # Display the frame
    cv2.imshow("Defect Detection - YOLOv8", annotated_frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
