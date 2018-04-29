import cv2
from PIL import Image as imaage
from PIL import ImageTk
from tkinter import *
from objDecMod import *

cap = cv2.VideoCapture("testVideo1.mp4")
#cap.set(cv2.CAP_PROP_FPS, 200)
while not cap.isOpened():
    cap = cv2.VideoCapture("testVideo1.mp4")
    #cap.set(cv2.CAP_PROP_FPS, 200)
    cv2.waitKey(1000)
    print("Wait for the header")

pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)

count=0

def updateImg():
    global count

    for i in range(0,250):
        count = count + 1
        flag, frame = cap.read()

    if not flag:
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos_frame-1)
        print("frame is not ready")
        # It is better to wait for a while for the next frame to be ready
        cv2.waitKey(1000)

    if cv2.waitKey(10) == 27:
        app.destroy()
        return

    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        # If the number of captured frames is equal to the total number of frames,
        # we stop
        app.destroy()
        return
    """
    if(flag == 0):
        app.destroy()
        return
    """

    count = count +1


    image_np = frame
    image_np_expanded = np.expand_dims(image_np, axis=0)
    # Actual detection.
    print("\n\n\nyess1\n\n\\n")
    output_dict = run_inference_for_single_image(image_np, detection_graph)
    print("\n\n\nyess2\n\n\\n")
    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks'),
        use_normalized_coordinates=True,
        line_thickness=8)


    image = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    #Resizing makes the frame not ready
    #image = cv2.resize(image, (0, 0), fx=0.6, fy=0.6)
    image = imaage.fromarray(image)
    img = ImageTk.PhotoImage(image)



    l.configure(image=img)
    l.image = img
    app.after(1, lambda: updateImg())



app = Tk()
app.title("Video Classifier")


l = Label(app, image=None)
l.pack()
app.after(1, lambda: updateImg())
app.mainloop()

