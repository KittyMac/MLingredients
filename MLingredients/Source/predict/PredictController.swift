import UIKit
import PlanetSwift
import CoreML
import Vision

class PredictController: PlanetViewController, CameraCaptureHelperDelegate {
    
    var captureHelper = CameraCaptureHelper(cameraPosition: .back)
    var model:VNCoreMLModel? = nil
    var overrideImage:CIImage? = nil
    let ciContext = CIContext(options: [:])
    var lastOriginalImage:CIImage? = nil

    
    func playCameraImage(_ cameraCaptureHelper: CameraCaptureHelper, image: CIImage, originalImage: CIImage, frameNumber:Int, fps:Int) {
        if frameNumber > 10 && frameNumber % 200 == 0 {
            if overrideImage != nil {
                lastOriginalImage = overrideImage!
            } else {
                lastOriginalImage = originalImage
            }
            
            //DispatchQueue.main.async {
            //    self.preview.imageView.image = UIImage(ciImage: self.lastOriginalImage!)
            //}
            
            
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
                        
                        
                        var numberOfWords = 0
                        for textObservation in results {
                            var numberOfCharacters = 0
                            for rectangleObservation in textObservation.characterBoxes! {
                                
                                if numberOfWords == 0 && numberOfCharacters == 0 {
                                    let w = self.lastOriginalImage!.extent.width
                                    let h = self.lastOriginalImage!.extent.height
                                    
                                    let perspectiveImagesCoords = [
                                        "inputTopLeft":CIVector(x:rectangleObservation.topLeft.x * w, y: rectangleObservation.topLeft.y * h),
                                        "inputTopRight":CIVector(x:rectangleObservation.topRight.x * w, y: rectangleObservation.topRight.y * h),
                                        "inputBottomLeft":CIVector(x:rectangleObservation.bottomLeft.x * w, y: rectangleObservation.bottomLeft.y * h),
                                        "inputBottomRight":CIVector(x:rectangleObservation.bottomRight.x * w, y: rectangleObservation.bottomRight.y * h),
                                        ]
                                    
                                    DispatchQueue.main.async {
                                        print(perspectiveImagesCoords)
                                        let extractedImage = self.lastOriginalImage!.applyingFilter("CIPerspectiveCorrection", parameters: perspectiveImagesCoords)
                                        print(extractedImage)
                                        self.preview.imageView.image = UIImage(ciImage: extractedImage)
                                    }
                                }
                                
                                numberOfCharacters += 1
                            }
                            numberOfWords += 1
                        }
                        
                        print("\(numberOfWords)")
                        
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
    
    func loadModel() {
        do {
            let modelURL = URL(fileURLWithPath: String(bundlePath:"bundle://Assets/predict/ingredients.mlmodel"))
            let compiledUrl = try MLModel.compileModel(at: modelURL)
            let model = try MLModel(contentsOf: compiledUrl)
            self.model = try? VNCoreMLModel(for: model)
        } catch {
            print(error)
        }
    }
    
    
    var currentOverrideImageIndex = 1
    
    override func viewDidLoad() {
        title = "Predict"
        mainBundlePath = "bundle://Assets/predict/predict.xml"
        loadView()
        
        overrideImage = CIImage(contentsOf: URL(fileURLWithPath: String(bundlePath: "bundle://Assets/predict/debug/IMG_0026.JPG")))
        
        if(overrideImage != nil) {
            navigationItem.rightBarButtonItem = UIBarButtonItem(title: "Next", style: .plain, target: self, action: #selector(nextOverrideImage))
        }

        super.viewDidLoad()
        
        // this is slightly complex so here it goes:
        // 1. the camera helper should be running barebones by itself
        // 2. we should start another thread which is responsible for processing a single image
        // 3. when that thread is done it waits for the camera helper to supply a new image, the processes that image
        
        captureHelper.delegate = self
        captureHelper.delegateWantsPlayImages = true
        
        UIApplication.shared.isIdleTimerDisabled = true
        
        loadModel()
        
        // debug to save the current image
        let tap = UITapGestureRecognizer(target: self, action: #selector(self.SaveImages(_:)))
        preview.view.addGestureRecognizer(tap)
        preview.view.isUserInteractionEnabled = true
    }
    
    @objc func nextOverrideImage() {
        currentOverrideImageIndex += 1
        if currentOverrideImageIndex > 17 {
            currentOverrideImageIndex = 0
        }
        let filePath = String(format:"bundle://Assets/predict/debug/IMG_%04d.JPG", currentOverrideImageIndex)
        overrideImage = CIImage(contentsOf: URL(fileURLWithPath: String(bundlePath: filePath)))
    }
    
    @objc func SaveImages(_ sender: UITapGestureRecognizer) {
         let originalImage = ciContext.createCGImage(lastOriginalImage!, from: lastOriginalImage!.extent)
         UIImageWriteToSavedPhotosAlbum(UIImage(cgImage: originalImage!), self, #selector(self.image(_:didFinishSavingWithError:contextInfo:)), nil)
    }
    
    @objc func image(_ image: UIImage, didFinishSavingWithError error: Error?, contextInfo: UnsafeRawPointer) {
        if let error = error {
            // we got back an error!
            let ac = UIAlertController(title: "Save error", message: error.localizedDescription, preferredStyle: .alert)
            ac.addAction(UIAlertAction(title: "OK", style: .default))
            present(ac, animated: true)
        } else {
            let ac = UIAlertController(title: "Saved!", message: "Your image has been saved to your photos.", preferredStyle: .alert)
            ac.addAction(UIAlertAction(title: "OK", style: .default))
            present(ac, animated: true)
        }
    }
    
    override func viewDidDisappear(_ animated: Bool) {
        UIApplication.shared.isIdleTimerDisabled = false
        captureHelper.stop()
    }
    
    fileprivate var preview: ImageView {
        return mainXmlView!.elementForId("preview")!.asImageView!
    }
    
}

