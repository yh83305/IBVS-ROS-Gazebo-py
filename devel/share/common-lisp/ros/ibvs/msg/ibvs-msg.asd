
(cl:in-package :asdf)

(defsystem "ibvs-msg"
  :depends-on (:roslisp-msg-protocol :roslisp-utils )
  :components ((:file "_package")
    (:file "DetectionResult" :depends-on ("_package_DetectionResult"))
    (:file "_package_DetectionResult" :depends-on ("_package"))
  ))