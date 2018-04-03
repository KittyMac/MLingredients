import UIKit
import PlanetSwift

extension UIView {
    func fillSuperview() {
        
        var constraintsView = superview
        while constraintsView != nil && constraintsView?.constraints.count == 0 {
            constraintsView = constraintsView!.superview
        }
        
        if constraintsView == nil {
            return
        }
        
        constraintsView!.addConstraint(NSLayoutConstraint(item: self, toItem: superview, equalAttribute: .width))
        constraintsView!.addConstraint(NSLayoutConstraint(item: self, toItem: superview, equalAttribute: .height))
        constraintsView!.addConstraint(NSLayoutConstraint(item: self, toItem: superview, equalAttribute: .left))
        constraintsView!.addConstraint(NSLayoutConstraint(item: self, toItem: superview, equalAttribute: .top))
        self.translatesAutoresizingMaskIntoConstraints = false
        superview!.translatesAutoresizingMaskIntoConstraints = false

    }
}
