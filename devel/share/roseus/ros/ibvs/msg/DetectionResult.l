;; Auto-generated. Do not edit!


(when (boundp 'ibvs::DetectionResult)
  (if (not (find-package "IBVS"))
    (make-package "IBVS"))
  (shadow 'DetectionResult (find-package "IBVS")))
(unless (find-package "IBVS::DETECTIONRESULT")
  (make-package "IBVS::DETECTIONRESULT"))

(in-package "ROS")
;;//! \htmlinclude DetectionResult.msg.html


(defclass ibvs::DetectionResult
  :super ros::object
  :slots (_s _Z ))

(defmethod ibvs::DetectionResult
  (:init
   (&key
    ((:s __s) (make-array 0 :initial-element 0.0 :element-type :float))
    ((:Z __Z) 0.0)
    )
   (send-super :init)
   (setq _s __s)
   (setq _Z (float __Z))
   self)
  (:s
   (&optional __s)
   (if __s (setq _s __s)) _s)
  (:Z
   (&optional __Z)
   (if __Z (setq _Z __Z)) _Z)
  (:serialization-length
   ()
   (+
    ;; float64[] _s
    (* 8    (length _s)) 4
    ;; float64 _Z
    8
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; float64[] _s
     (write-long (length _s) s)
     (dotimes (i (length _s))
       (sys::poke (elt _s i) (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
       )
     ;; float64 _Z
       (sys::poke _Z (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; float64[] _s
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _s (instantiate float-vector n))
     (dotimes (i n)
     (setf (elt _s i) (sys::peek buf ptr- :double)) (incf ptr- 8)
     ))
   ;; float64 _Z
     (setq _Z (sys::peek buf ptr- :double)) (incf ptr- 8)
   ;;
   self)
  )

(setf (get ibvs::DetectionResult :md5sum-) "4e77c3858d67058b1a7e3b743cd831cb")
(setf (get ibvs::DetectionResult :datatype-) "ibvs/DetectionResult")
(setf (get ibvs::DetectionResult :definition-)
      "float64[] s
float64 Z
")



(provide :ibvs/DetectionResult "4e77c3858d67058b1a7e3b743cd831cb")


