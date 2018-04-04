import UIKit
import PlanetSwift
import CoreML
import Vision

class MainController: PlanetViewController {
    
    override func viewDidLoad() {
        super.viewDidLoad()
        title = "Main"
        mainBundlePath = "bundle://Assets/main/main.xml"
        loadView()
        
        UIApplication.shared.isIdleTimerDisabled = true
        
        trainButton.button.add(for: .touchUpInside) {
            self.navigationController?.pushViewController(TrainController(), animated: true)
        }
        
        predictButton.button.add(for: .touchUpInside) {
            self.navigationController?.pushViewController(PredictController(), animated: true)
        }
    }
    fileprivate var predictButton: Button {
        return mainXmlView!.elementForId("predictButton")!.asButton!
    }
    fileprivate var trainButton: Button {
        return mainXmlView!.elementForId("trainButton")!.asButton!
    }

}

