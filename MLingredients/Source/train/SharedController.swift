import UIKit
import PlanetSwift
import CoreML
import Vision

class SharedController: PlanetViewController, CameraCaptureHelperDelegate {
    
    let ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ 1234567890,"
    
    var captureHelper = CameraCaptureHelper(cameraPosition: .back)
    var overrideImage:CIImage? = nil
    let ciContext = CIContext(options: [:])
    var lastOriginalImage:CIImage? = nil
    
    var lastImageUsedForObservations:CIImage? = nil
    var observationsToProcess:[Any?] = []
    
    
    func newObservationsAvailable () {
        // override in subclass
    }
    
    func newImageAvailable () {
        // override in subclass
    }
    
    func clearObservations() {
        observationsToProcess.removeAll()
        lastImageUsedForObservations = nil
    }

    
    func playCameraImage(_ cameraCaptureHelper: CameraCaptureHelper, image: CIImage, originalImage: CIImage, frameNumber:Int, fps:Int) {
        if frameNumber > 10 {
            
            
            if observationsToProcess.count == 0 {
                
                newImageAvailable()
                
                if overrideImage != nil {
                    lastOriginalImage = overrideImage!
                } else {
                    lastOriginalImage = originalImage
                }

                //let convertedImage = image |> adjustColors |> convertToGrayscale
                let handler = VNImageRequestHandler(ciImage: self.lastOriginalImage!)
                let request: VNDetectTextRectanglesRequest =
                    VNDetectTextRectanglesRequest(completionHandler: { [unowned self] (request, error) in
                        if (error != nil) {
                            print("Got Error In Run Text Dectect Request :(")
                        } else {
                            guard let results = request.results as? Array<VNTextObservation> else {
                                fatalError("Unexpected result type from VNDetectTextRectanglesRequest")
                            }
                            if (results.count == 0) {
                                return
                            }
                            
                            self.clearObservations()
                            
                            self.lastImageUsedForObservations = self.lastOriginalImage
                            
                            var avgAspect:CGFloat = 0
                            var numAspect:CGFloat = 0
                            for textObservation in results {
                                for rectangleObservation in textObservation.characterBoxes! {
                                    let charW = rectangleObservation.bottomRight.x - rectangleObservation.bottomLeft.x
                                    let charH = rectangleObservation.topRight.y - rectangleObservation.bottomRight.y
                                    let charAspect = charW/charH
                                    
                                    // sanity assumption for single character aspect
                                    if charAspect < 1.4 {
                                        avgAspect += charAspect
                                        numAspect += 1.0
                                    }
                                }
                            }
                            if numAspect == 0 {
                                self.clearObservations()
                                return
                            }
                            
                            avgAspect /= numAspect
                            
                            
                            for textObservation in results {
                                for rectangleObservation in textObservation.characterBoxes! {
                                    
                                    // TODO: try and detect when the vision detected more than one character in
                                    // a box. We can try this simply by checking what the aspect ratio of the ractangle
                                    // is; if its too wide perhaps it is 2 or 3 characters in the box
                                    
                                    let w = self.lastOriginalImage!.extent.width
                                    let h = self.lastOriginalImage!.extent.height
                                    
                                    
                                    let charW = rectangleObservation.bottomRight.x - rectangleObservation.bottomLeft.x
                                    let charH = rectangleObservation.topRight.y - rectangleObservation.bottomRight.y
                                    let charAspect = charW/charH
                                    
                                    var subdivisions = Int(round(charAspect / avgAspect))
                                    
                                    subdivisions = 1
                                    
                                    //print("\(subdivisions) \(charAspect)        \(avgAspect)")
                                    
                                    var charX = rectangleObservation.bottomLeft.x
                                    
                                    let deltaW = charW / CGFloat(subdivisions)
                                    
                                    for _ in 0..<subdivisions {
                                        
                                        let perspectiveImagesCoords = [
                                            "inputTopLeft":CIVector(x:(charX + 0) * w, y: rectangleObservation.topLeft.y * h),
                                            "inputTopRight":CIVector(x:(charX + deltaW) * w, y: rectangleObservation.topRight.y * h),
                                            "inputBottomLeft":CIVector(x:(charX + 0) * w, y: rectangleObservation.bottomLeft.y * h),
                                            "inputBottomRight":CIVector(x:(charX + deltaW) * w, y: rectangleObservation.bottomRight.y * h),
                                            ]
                                        
                                        self.observationsToProcess.append(perspectiveImagesCoords)
                                        
                                        charX += deltaW
                                    }
                                }
                                
                                self.observationsToProcess.append(nil)
                            }
                            
                            DispatchQueue.main.async {
                                self.newObservationsAvailable()
                            }
                        }
                    })
                request.reportCharacterBoxes = true
                do {
                    try handler.perform([request])
                } catch {
                    print(error)
                }
                
            }

            
        }
    }
    
    var currentOverrideImageIndex = 0
    
    override func viewDidLoad() {
        overrideImage = CIImage(contentsOf: URL(fileURLWithPath: String(bundlePath: "bundle://Assets/predict/debug/IMG_0042.JPG")))
        
        if(overrideImage != nil) {
            navigationItem.rightBarButtonItem = UIBarButtonItem(title: "Next", style: .plain, target: self, action: #selector(nextOverrideImage))
        }

        captureHelper.delegate = self
        captureHelper.delegateWantsPlayImages = true
    }
    
    @objc func nextOverrideImage() {
        
        do {
            let debugUrl = URL(fileURLWithPath: String(bundlePath: "bundle://Assets/predict/debug/"))

            let fileURLs = try FileManager.default.contentsOfDirectory(at: debugUrl, includingPropertiesForKeys: nil)
            
            currentOverrideImageIndex += 1
            if currentOverrideImageIndex >= fileURLs.count {
                currentOverrideImageIndex = 0
            }
            
            overrideImage = CIImage(contentsOf: fileURLs[currentOverrideImageIndex])
            
        } catch {
            print("Error going to next debug image")
        }
        
        clearObservations()
    }
    
    override func viewDidDisappear(_ animated: Bool) {
        captureHelper.stop()
    }
}

