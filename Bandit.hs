module Bandit 
  ( -- Model
    Bandit(..), mkBandit
    -- Solution algorithms
  , gittinsIndex, gittinsIndexAt  
  ) where

import qualified Numeric.LinearAlgebra as N
import qualified Data.Vector as V
import qualified MDP as MDP

data Bandit = Bandit
  { stateSize  :: !Int
  , reward     :: N.Vector Double
  , transition :: N.Matrix Double
  }

type Discount = Double

mkBandit :: Int -> [Double] -> [[Double]] -> Bandit
mkBandit stateSize_ reward_ transition_ = 
  Bandit stateSize_ (N.fromList reward_) (N.fromLists transition_)

-- Use VWB algorithm to calculate Gittins index
gittinsIndex :: Discount -> Bandit -> N.Vector Double
gittinsIndex = undefined


-- Use Katehakis and Veinott's "restart in x" algorithm to calculate Gittins index
gittinsIndexAt :: Discount -> Bandit -> Int -> Double
gittinsIndexAt β (Bandit s r p)  state = (1-β) * (value N.@> state) where
    mdp :: MDP.MDP
    mdp = MDP.MDP s
                  2 -- 1st action is continue, 2nd action is restart
                  (V.fromList [r, N.konst (r N.@> state)  s])
                  (V.fromList [p, N.fromRows (replicate s (MDP.takeRow state p))])
    (value, _) = last $ MDP.policyIteration β mdp

