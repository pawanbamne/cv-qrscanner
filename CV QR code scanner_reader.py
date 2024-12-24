import cv2

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Create a QRCodeDetector object
qr_detector = cv2.QRCodeDetector()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert the frame to grayscale for better contrast detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to increase contrast (optional)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    # Detect and decode QR code from the processed frame
    data, bbox, _ = qr_detector.detectAndDecode(thresh)

    # If QR code is detected, display the decoded data and draw the bounding box
    if bbox is not None and len(bbox) > 0:
        # Convert the bbox points to integers for drawing
        bbox = bbox.astype(int)

        # Draw the bounding box around the QR code
        for i in range(len(bbox)):
            point1 = tuple(bbox[i][0])
            point2 = tuple(bbox[(i + 1) % len(bbox)][0])
            cv2.line(frame, point1, point2, color=(255, 0, 0), thickness=2)

        # Display the decoded data
        if data:
            cv2.putText(frame, f"QR Code Data: {data}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    else:
        # No QR code detected or invalid bounding box
        cv2.putText(frame, "QR Code not detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the resulting frame with bounding box and decoded data
    cv2.imshow("QR Code Scanner", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
