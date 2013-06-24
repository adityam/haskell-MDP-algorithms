module MDP 
  ( -- Model
    MDP (..) , mkMDP
    -- Types
  , QFunction , VFunction , Policy , ValuePolicyPair
  , qUpdate , vUpdate , bellmanUpdate 
  , policyMDP
  , valueIteration , valueIterationN
  , policyIteration, policyIterationN
  , spanNorm
  ) where

import qualified Data.Vector as V
import qualified Numeric.LinearAlgebra as N
import Control.Arrow ( (&&&), (***) )

import Data.Function (on)

-- Data type to model a finite Markov Decision Process (MDP). 
--
-- Let $S = {\tt stateSize}$ and $A = {\tt actionSize}$ denote the size of the
-- state and action spaces respectively. 
--
-- `reward` is a $S×A$ matrix, and `transition` is a collection of $A$
-- matrices, each of size $S×S$.

data MDP = MDP 
   { stateSize  :: !Int
   , actionSize :: !Int
   , reward     :: V.Vector (N.Vector Double)
   , transition :: V.Vector (N.Matrix Double)
   }

mkMDP :: Int -> Int -> [[Double]] -> [[[Double]]] -> MDP
mkMDP stateSize_ actionSize_ reward_ transition_ = 
    MDP stateSize_ actionSize_ 
        (V.fromList . map N.fromList  $ reward_)
        (V.fromList . map N.fromLists $ transition_)

-- TODO:
-- check :: Function to check if the model is correct

type QFunction = N.Matrix Double
type VFunction = N.Vector Double
type Policy    = V.Vector Int

type ValuePolicyPair = (VFunction, Policy)

type Discount = Double

-- In vectorized form, the Q-update is given by
-- 
-- $$Q(⋅,u) = r(⋅,u) + β P(u) V(⋅)$$
-- 
-- We micro-optimize the implementation by rearranging terms as
--
-- $$Q(⋅,u) = r(⋅,u) + P(u) (β V(⋅))$$
--
-- In the following code, the second term of the summand is called
-- `expectedCostToGo`. 
--
-- All collection of vectors and matrices are stored as `Data.Vector`. The hope
-- is that this will make it easy to parallelize the algorithm in the future.
--
-- The final result is converted to a `Numeric.LinearAlgebra.Matrix` because we
-- later use the `maxElement` and `maxIndex` functions from
-- `Numeric.LinearAlgebra`.

{-# INLINE qUpdate #-}
qUpdate :: Discount -> MDP -> ValuePolicyPair -> QFunction
qUpdate β mdp (vFunction,_) = N.fromColumns . V.toList $ qMatrix
    where
    rFunction = reward mdp     -- The collection of vectors  $\{ r(⋅, u) | u \in A \}$
    pMatrices = transition mdp -- The collection of matrices $\{ P(u) | u \in A \}$

    scaledVFunction = N.scale β vFunction -- See note about micro-optimization above

    -- TODO: Parallelize the computation of expectedCostToGo
    expectedCostToGo = V.map (N.<> scaledVFunction) pMatrices -- This is the second term of the summand
    qMatrix  = V.zipWith (+) rFunction expectedCostToGo

-- Value update is given by
-- 
-- $$ V(x) = max \{ Q(x,u) : u \in U \} $$

{-# INLINE vUpdate #-}
vUpdate :: QFunction -> ValuePolicyPair
vUpdate  = (N.fromList *** V.fromList) . unzip . map (N.maxElement &&& N.maxIndex) . N.toRows

-- Bellman update is the composition of `vUpdate` and `qUpdate`.

{-# INLINE bellmanUpdate #-}
bellmanUpdate :: Discount -> MDP -> ValuePolicyPair -> ValuePolicyPair
bellmanUpdate β mdp = vUpdate . qUpdate β mdp 


-- Construct a MDP corresponding to a single policy 

-- {-# INLINE policyMDP #-}
policyMDP :: Discount -> MDP -> Policy -> MDP
policyMDP β mdp policy =
    let
       units :: V.Vector (N.Vector Double)
       units   = V.fromList . N.toRows . N.ident $ stateSize mdp

       chosenReward :: V.Vector (N.Vector Double)
       chosenReward = V.backpermute (reward mdp) policy

       -- We can find the rFunction by matrix multilincation, but it is cheaper
       -- to select elements from a vector
       --
       --       rFunction = V.zipWith N.dot units chosenReward
       --
       rFunction :: V.Vector Double
       rFunction = V.imap (flip (N.@>)) chosenReward

       chosenMatrices :: V.Vector (N.Matrix Double)
       chosenMatrices = V.backpermute (transition mdp) policy

       -- We can find the pMatrix by matrix multilincation, but it is cheaper
       -- to select rows from a matrix
       --
       --       pMatrix = V.zipWith (N.<>) units chosenMatrices
       --
       pMatrix :: V.Vector (N.Vector Double)
       pMatrix = V.imap takeRow chosenMatrices

    in mdp { reward     = V.singleton . N.fromList . V.toList $ rFunction 
           , transition = V.singleton . N.fromRows . V.toList $ pMatrix 
           }
  
{-# INLINE policyUpdate #-}
policyUpdate :: Discount -> MDP -> ValuePolicyPair -> ValuePolicyPair 
policyUpdate β mdp vp = 
    let
        -- Find best response policy
        (_, p2) = bellmanUpdate β mdp vp

        --  Find performance of best response policy using
        --  $(I - β P(d))^{-1} r(d)$
        mdp'    = policyMDP β mdp p2
        matrix  = N.ident (stateSize mdp') - N.scale β (V.head . transition $ mdp')
        -- In general, using linearSolve A B is more stable than calculating A^{-} B
        -- v2      = N.inv matrix N.<> (V.head . reward $ mdp') 
        v2      = matrix N.<\> (V.head . reward $ mdp')
    in (v2, p2)

-- Start with the all zeros vector and run `bellmanUpdate` until the `spanNorm`
-- of successive terms with within $ε(1-β)/β$ of each other. This ensures that
-- the renormalized value function is within $ε$ of the optimal solution.

-- To stop the computation after n iterations, use
--
--      take n . valueIteration ε β $ mdp
--
-- or (to see the iteration count)
--
--      take n . valueIterationN ε β $ mdp

valueIteration :: Double -> Discount -> MDP -> [ValuePolicyPair]
valueIteration ε β mdp = 
    let 
        k = if β < 1 
               then (1-β)/β
               else 1
        δ = k*ε

        initial :: ValuePolicyPair 
        initial = (N.konst 0 (stateSize mdp), V.replicate (actionSize mdp) 0)

        bellmanUpdate' :: ValuePolicyPair -> ValuePolicyPair
        bellmanUpdate' = bellmanUpdate β mdp

        iterations :: [ValuePolicyPair]
        iterations = iterate bellmanUpdate' initial

        pairedIteration :: [(ValuePolicyPair, ValuePolicyPair)]
        pairedIteration = zip iterations (tail iterations)

        stoppingRule :: ValuePolicyPair -> ValuePolicyPair -> Bool
        stoppingRule (v1,_) (v2,_) = spanNorm v1 v2 < δ

        -- See Puterman 6.6.12 for details
        renormalize :: (ValuePolicyPair, ValuePolicyPair) -> ValuePolicyPair
        renormalize ( (v1, _), (v2, p2) ) = let w = N.maxElement (v2-v1)
                                            in (v2 + N.scalar (w/k), p2)

     in map renormalize . takeWhile (not . uncurry stoppingRule) $ pairedIteration

-- Value iteration that keeps track of the iteration count as well.
valueIterationN:: Double -> Discount-> MDP -> [(Int, ValuePolicyPair)]
valueIterationN ε β mdp = zip [2..] (valueIteration ε β mdp)

policyIteration :: Discount -> MDP -> [ValuePolicyPair]
policyIteration β mdp = 
    let 
        initial :: ValuePolicyPair 
        initial = (N.konst 0 (stateSize mdp), V.replicate (actionSize mdp) 0)

        policyUpdate' :: ValuePolicyPair -> ValuePolicyPair
        policyUpdate' = policyUpdate β mdp

        iterations :: [ValuePolicyPair]
        iterations = iterate policyUpdate' initial

        pairedIteration :: [(ValuePolicyPair, ValuePolicyPair)]
        pairedIteration = zip iterations (tail iterations)

        stoppingRule :: ValuePolicyPair -> ValuePolicyPair -> Bool
        -- stoppingRule ((_,p1), (_,p2)) = p1 == p2
        stoppingRule = (==) `on` snd

     in map snd . takeWhile' (not . uncurry stoppingRule) $ pairedIteration

-- Value iteration that keeps track of the iteration count as well.
policyIterationN:: Discount -> MDP -> [(Int, ValuePolicyPair)]
policyIterationN β mdp = zip [2..] (policyIteration β mdp)

-- Helper functions

{-# INLINE spanNorm #-}
spanNorm :: N.Vector Double -> N.Vector Double -> Double
spanNorm u v = N.maxElement w - N.minElement w
      where w = u - v

{-# INLINE takeRow #-}
takeRow :: N.Element e => Int -> N.Matrix e -> N.Vector e
takeRow idx mat = N.flatten . N.subMatrix (idx,0) (1, N.cols mat) $ mat

-- Modified version of takeWhile

takeWhile'               :: (a -> Bool) -> [a] -> [a]
takeWhile' _ []          =  []
takeWhile' p (x:xs) 
             | p x       =  x : takeWhile' p xs
             | otherwise =  x : [] 

