package org.apache.mahout.math.algorithms

import org.apache.mahout.math.algorithms.Model
import org.apache.mahout.math.algorithms.regression.{GlmModel, LinearRegressorModel}
import org.apache.mahout.math.drm._
import org.apache.mahout.math.drm.RLikeDrmOps._
import org.apache.mahout.math.scalabindings._
import org.apache.mahout.math.scalabindings.RLikeOps._
import org.apache.mahout.test.DistributedMahoutSuite
import org.scalatest.{FunSuite, Matchers}

trait GlmSuiteBase extends DistributedMahoutSuite with Matchers {
  this: FunSuite =>

  test("ordinary least squares using glm") {
    /*
    R Prototype:
    dataM <- matrix( c(2, 2, 10.5, 10, 29.509541,
      1, 2, 12,   12, 18.042851,
      1, 1, 12,   13, 22.736446,
      2, 1, 11,   13, 32.207582,
      1, 2, 12,   11, 21.871292,
      2, 1, 16,   8,  36.187559,
      6, 2, 17,   1,  50.764999,
      3, 2, 13,   7,  40.400208,
      3, 3, 13,   4,  45.811716), nrow=9, ncol=5, byrow=TRUE)


    X = dataM[, c(1,2,3,4)]
    y = dataM[, c(5)]

    model <- lm(y ~ X )
    summary(model)

     */

    val drmData = drmParallelize(dense(
      (2, 2, 10.5, 10, 29.509541),  // Apple Cinnamon Cheerios
      (1, 2, 12,   12, 18.042851),  // Cap'n'Crunch
      (1, 1, 12,   13, 22.736446),  // Cocoa Puffs
      (2, 1, 11,   13, 32.207582),  // Froot Loops
      (1, 2, 12,   11, 21.871292),  // Honey Graham Ohs
      (2, 1, 16,   8,  36.187559),  // Wheaties Honey Gold
      (6, 2, 17,   1,  50.764999),  // Cheerios
      (3, 2, 13,   7,  40.400208),  // Clusters
      (3, 3, 13,   4,  45.811716)), numPartitions = 2)


    val drmX = drmData(::, 0 until 4)
    val drmY = drmData(::, 4 until 5)

    val model:LinearRegressorModel[Int] = new GlmModel[Int]().apply(drmX, drmY,2,3,null)

    val estimate = model.beta
    val Ranswers = dvec(-1.336265, -13.157702, -4.152654, -5.679908, 163.179329)

    val epsilon = 1E-6
    (estimate - Ranswers).sum should be < epsilon

    // TODO add test for S.E / pvalue
  }




}
