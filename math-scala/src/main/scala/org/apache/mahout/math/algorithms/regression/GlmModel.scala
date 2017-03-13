package org.apache.mahout.math.algorithms.regression


import org.apache.mahout.math.algorithms.{Fitter, Model}
import org.apache.mahout.math.algorithms.regression.tests.FittnessTests
import org.apache.mahout.math.drm._
import org.apache.mahout.math.drm.DrmLike
import org.apache.mahout.math.drm.RLikeDrmOps._
import org.apache.mahout.math.scalabindings.dvec
import org.apache.mahout.math.{Matrix, Vector => MahoutVector}
import org.apache.mahout.math.scalabindings.RLikeOps._

import scala.language.higherKinds

/*** Keeping these comments here for now to help us organize this effort
 * Train a GLM model. The types of model are given by the values of opts.links (IMat). They are:
 - 0 = linear model (squared loss)
 - 1 = Ordinary least squares model
 - 2 = CochraneOrcutt model
 - 3 = LogisticRegression model
 - 4 = Support Vector Machines nodel
 */
class GlmModel[K] extends LinearRegressorModel[K] {

  // Create a set of apply functions for each of the types
  // of models we want to process, need to rethink how to make
  //this completely generic and how to make it work with the other
  //functions that are available in the fitter

  def predict(drmPredictors: DrmLike[K]): DrmLike[K] = {
    return null
  }

  //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  /** Build a Linear Regressor model through the apply function
    * Inputs specify a set number of rows but have constrained number of columns
    *  @param hyperparameters all of the hyperparameters for linear regression
    *  @param iterations the number of iterations
    *  @return A model in this case the LinearRegressorFitter
    */
  def apply (drmFeatures:DrmLike[K],drmTarget:DrmLike[K],algorithm:Int,iterations:Int,hyperparameters: (Symbol, Any)*): LinearRegressorModel[K] =
  {
    if (algorithm==1)
    {
      var regressor: LinearRegressorFitter[K] = hyperparameters.asInstanceOf[Map[Symbol,
        LinearRegressorFitter[K]]].getOrElse('regressor, new OrdinaryLeastSquares[K]())
      var regressionModel: LinearRegressorModel[K] = regressor.fit(drmFeatures, drmTarget)
      regressionModel
    } else {
      val model = new OrdinaryLeastSquares[K]().fit(drmFeatures, drmTarget)
      model
    }
  } // apply
}


trait GlmFitter[K] extends RegressorFitter[K] {

  var addIntercept: Boolean = _
  var calcStandardErrors: Boolean = _
  var calcCommonStatistics: Boolean = _

  def fit(drmX: DrmLike[K],
          drmTarget: DrmLike[K],
          hyperparameters: (Symbol, Any)*): GlmModel[K]


  def setStandardHyperparameters(hyperparameters: Map[Symbol, Any] = Map('foo -> None)): Unit = {
    calcCommonStatistics = hyperparameters.asInstanceOf[Map[Symbol, Boolean]].getOrElse('calcCommonStatistics, true)
    calcStandardErrors = hyperparameters.asInstanceOf[Map[Symbol, Boolean]].getOrElse('calcStandardErrors, true)
    addIntercept = hyperparameters.asInstanceOf[Map[Symbol, Boolean]].getOrElse('addIntercept, true)
  }




  def calculateStandardError[M[K] <: GlmModel[K]](X: DrmLike[K],
                                                  drmTarget: DrmLike[K],
                                                  drmXtXinv: Matrix,
                                                  model: M[K]): M[K] = {
    import org.apache.mahout.math.function.Functions.SQRT
    import org.apache.mahout.math.scalabindings.MahoutCollections._
    var modelOut = model
    val yhat = X %*% model.beta
    val residuals = drmTarget - yhat
    val ete = (residuals.t %*% residuals).collect // 1x1
    val n = drmTarget.nrow
    val k = safeToNonNegInt(X.ncol)
    val invDegFreedomKindOf = 1.0 / (n - k)
    val varCovarMatrix = invDegFreedomKindOf * ete(0,0) * drmXtXinv
    val se = varCovarMatrix.viewDiagonal.assign(SQRT)
    val tScore = model.beta / se
    val tDist = new org.apache.commons.math3.distribution.TDistribution(n-k)
    val pval = dvec(tScore.toArray.map(t => 2 * (1.0 - tDist.cumulativeProbability(t)) ))
    //degreesFreedom = k
    modelOut.summary = "Coef.\t\tEstimate\t\tStd. Error\t\tt-score\t\t\tPr(Beta=0)\n" +
      (0 until k).map(i => s"X${i}\t${model.beta(i)}\t${se(i)}\t${tScore(i)}\t${pval(i)}").mkString("\n")

    modelOut.se = se
    modelOut.tScore = tScore
    modelOut.pval = pval
    modelOut.degreesFreedom = X.ncol

    if (calcCommonStatistics){
      modelOut = calculateCommonStatistics(modelOut, drmTarget, residuals)
    }
    modelOut
  }

  def calculateCommonStatistics[M[K] <: GlmModel[K]](model: M[K],
                                                     drmTarget: DrmLike[K],
                                                     residuals: DrmLike[K]): M[K] ={
    var modelOut = model
    modelOut = FittnessTests.CoefficientOfDetermination(model, drmTarget, residuals)
    modelOut = FittnessTests.MeanSquareError(model, residuals)
    modelOut
  }

  def modelPostprocessing[M[K] <: GlmModel[K]](model: M[K],
                                               X: DrmLike[K],
                                               drmTarget: DrmLike[K],
                                               drmXtXinv: Matrix): M[K] = {
    var modelOut = model
    if (calcStandardErrors) {
      modelOut = calculateStandardError(X, drmTarget, drmXtXinv, model )
    } else {
      modelOut.summary = "Coef.\t\tEstimate\n" +
        (0 until X.ncol).map(i => s"X${i}\t${modelOut.beta(i)}").mkString("\n")
      if (calcCommonStatistics) { // we do this in calcStandard errors to avoid calculating residuals twice
      val residuals = drmTarget - (X %*% modelOut.beta)
        modelOut = calculateCommonStatistics(modelOut, drmTarget, residuals)
      }

      modelOut
    }

    if (addIntercept) {
      model.summary.replace(s"X${X.ncol - 1}", "(Intercept)")
    }
    model
  }
}

