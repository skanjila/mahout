import org.apache.mahout.math.algorithms.regression.OrdinaryLeastSquares
import org.apache.mahout.math.drm._
import org.apache.mahout.math.drm.RLikeDrmOps._
import org.apache.mahout.math.scalabindings._
import org.apache.mahout.math.scalabindings.RLikeOps._
import org.apache.mahout.test.DistributedMahoutSuite
import org.scalatest.{FunSuite, Matchers}

trait GlmSuiteBase extends DistributedMahoutSuite with Matchers {
  this: FunSuite =>

  test("linear regression") {
  }

  test("logistic regression") {
  }



}
