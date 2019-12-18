import System.IO(readFile, openFile, hClose, hFlush, hPutStrLn, stdout, IOMode( WriteMode ))
import System.Exit(die)
import Data.Char(toLower, ord)
import System.Environment(getArgs, getProgName)
import Data.Map(Map, toList, fromList, fromListWith, unionsWith)
import Data.List(sortBy, isPrefixOf)
import Control.Monad(forever)
import Control.DeepSeq(deepseq)

-- stack install vector
import Data.Vector as Vector (Vector, create, unsafeIndex)
import Data.Vector.Mutable as MVector (replicate, read, write)

-- stack install split
import Data.List.Split(chunksOf) 

-- stack install parallel
import Control.Parallel.Strategies

-- stack install parallel-io
import Control.Concurrent.ParallelIO.Global(parallel_, stopGlobalPool) 
-- Ref: http://hackage.haskell.org/package/parallel-io-0.3.3/docs/Control-Concurrent-ParallelIO-Global.html

import System.CPUTime(getCPUTime)
-- Ref: https://wiki.haskell.org/Timing_computations

import System.FilePath.Posix(takeFileName)
-- Ref: http://hackage.haskell.org/package/filepath-1.4.2.1/docs/System-FilePath-Posix.html

import System.Directory(listDirectory)
-- Ref: https://hackage.haskell.org/package/directory-1.3.4.0/docs/System-Directory.html

chunkSize :: Int
chunkSize = 40000

unionFanIn :: Int
unionFanIn = 8

hashSize :: Int
hashSize = 26

lexiconDir :: String
lexiconDir = "../lexicon/"

maxSuggest :: Int
maxSuggest = 10

main :: IO ()
main = do args <- getArgs
          case args of 
            ["analyze", filename] -> analyzeCorpus filename
            ["query", string] -> singleQuery string
            ["cli"] -> cli
            _ -> do pn <- getProgName
                    putStrLn $ "Usage: " 
                    putStrLn $ "\tAnalyze corpus\t--\t" ++ pn ++ " analyze <filename>"
                    putStrLn $ "\tSingle query\t--\t" ++ pn ++ " query <string> "
                    putStrLn $ "\tInteractive CLI\t--\t" ++ pn ++ " cli"
                    die $ ""

-- Analyze corpus using trigram and save the result for future queries.
analyzeCorpus :: FilePath -> IO ()
analyzeCorpus inputFileName = do
    corpus <- readFile inputFileName
    let triGram = makeTriGram $ cleanContent corpus
    let mapJobs = chunksOf chunkSize triGram
    let mrMatrix = map doMap mapJobs `using` parList rdeepseq
    let reduceJobs = map (getCol mrMatrix) [0..hashSize-1]
    let lexicon = map doReduce reduceJobs `using` parList rseq

    parallel_ $ map (doSave inputFileName) $ zip [0..hashSize-1] lexicon
    putStrLn $ "Processed lexicon size: " ++ (show $ sum $ map length lexicon)
    stopGlobalPool

singleQuery :: [Char] -> IO ()
singleQuery s = do
    let biGram = reverse $ take 2 $ reverse $ cleanContent s
    if length biGram < 2 then do 
        putStrLn $ "String (" ++ show s ++ ") can not form a biGram." 
        return ()
    else do
        lexicon <- loadLexicon $ hashNGram (head biGram, last biGram, "")
        
        query lexicon biGram

cli :: IO a
cli = do
    putStrLn $ "Initializing lexicon..."
    start <- getCPUTime
    lexicons <- sequence $ map loadLexicon [0..hashSize-1]
    deepseq lexicons $ putStrLn "Initialization accomplished!"
    end   <- getCPUTime
    let diff = (fromIntegral (end - start)) / (10^(12::Int))
    putStrLn $ "Successfully loaded " ++ ( show $ sum $ map length lexicons ) ++ " trigrams in " ++ (show (diff :: Double)) ++ " seconds."

    forever $ do
        putStr $ "Hasgole> "
        hFlush stdout
        s <- getLine

        let biGram = reverse $ take 2 $ reverse $ cleanContent s
        if length biGram < 2 then do 
            putStrLn $ "String (" ++ show s ++ ") can not form a biGram."
        else do
            let lexicon = lexicons !! hashNGram (head biGram, last biGram, "")
            query lexicon biGram

-- Return a list of cleaned words from the content.
cleanContent :: String -> [[Char]]
cleanContent c = filter (not.null) $ map cleanWord $ words c
    where 
        cleanWord :: String -> String
        cleanWord "" = ""
        cleanWord (x:xs) | isLetter  = toLower x : cleanWord xs
                         | isValid   = x : cleanWord xs
                         | otherwise = cleanWord xs
            where
                isLetter = (ascii >= 65 && ascii <=90) || (ascii >= 97 && ascii <=122)
                ascii = ord x
                isValid = x `elem` ['-', '\'']

-- Given a list of words, combine continuous 3 words as a trigram.
makeTriGram :: [a] -> [(a, a, a)]
makeTriGram w1 = zipWith3 ((,,)) w1 w2 w3
    where 
        w2 = tail w1
        w3 = tail w2

-- Get Hash Value of a trigram
hashNGram :: ([Char], b, c) -> Int
hashNGram (k1, _, _) = rawHashVal k1 `mod` hashSize
    where 
        rawHashVal s | isValid s = ascii s - 97
                     | otherwise = 0
            where 
                isValid w = length w > 0 && ascii w >= 97 && ascii w <= 122
                ascii k = ord $ head k

-- Assign values to the slot determined by the hash function.
assignWith :: Foldable t => (a -> Int) -> t (a, b) -> Vector [(a, b)]
assignWith hashFunc triGramWithFreq = Vector.create $ do
    vec <- MVector.replicate hashSize []
    let addGram e@(g, _) = do
        let i = hashFunc g
        es <- MVector.read vec i
        MVector.write vec i $ e : es
    
    Prelude.mapM_ addGram triGramWithFreq
    return vec
    

getCol :: [Vector [a]] -> Int -> [[a]]
getCol matrix col = foldr (:) [[]] $ map (\row -> unsafeIndex row col) matrix

doMap :: [([Char], b, c)] -> Vector [(([Char], b, c), Int)]
doMap xs = assignWith hashNGram $ map (\k -> (k, 1 :: Int)) xs

doReduce :: Ord a => [[(a, Int)]] -> [(a, Int)]
doReduce xs = reverse.sortKey $ toList $ unionMaps $ map (fromListWith (+)) xs

-- Recursively union maps in parallel.
unionMaps :: (Ord k, Num a) => [Map k a] -> Map k a
unionMaps ms | length ms <= unionFanIn = unionsWith (+) ms
             | otherwise               = unionMaps partiallyUnionedMaps
    where 
        partiallyUnionedMaps = map (unionsWith (+)) (chunksOf unionFanIn ms) `using` parList rseq

-- Sort kv pairs in ascending order by value.
sortKey :: [(a, Int)] -> [(a, Int)]
sortKey = sortBy (\(_,v1) (_,v2) -> v1 `compare` v2)

doSave :: (Show a1, Show a2) => [Char] -> (a2, [a1]) -> IO ()
doSave inputFileName (hashIndex, kvList) = do
    let outputFileName = lexiconDir ++ "lexicon_" ++ show hashIndex ++ "_" ++ takeFileName inputFileName
    h <- openFile outputFileName WriteMode
    mapM_ (hPutStrLn h) $ map show $ kvList
    hClose h
    return ()

query :: Eq a => [((a, a, [Char]), Int)] -> [a] -> IO ()
query lexicon biGram = do
    start <- getCPUTime
    let matchedResult = filter (\((k1,k2,_),_) -> head biGram == k1 && last biGram == k2) lexicon
    let printResult = take maxSuggest $ map (\((_,_,k3),v) -> show v ++ " " ++ k3) $ reverse $ sortKey matchedResult
    end   <- getCPUTime

    mapM_ putStrLn printResult
    let diff = (fromIntegral (end - start)) / (10^(12::Int))
    putStrLn $ "(Find " ++ ( show $ length printResult ) ++ " results in " ++ (show (diff :: Double)) ++ " seconds)"

getPrefix :: Int -> [Char]
getPrefix i | i >= 0 && i < hashSize = "lexicon_" ++ show i ++ "_"
            | otherwise              = "lexicon_"

loadLexicon :: Int -> IO [(([Char], [Char], [Char]), Int)]
loadLexicon hashIndex = do
    let prefix = getPrefix hashIndex

    files <- listDirectory lexiconDir
    let validFiles = filter (isPrefixOf prefix) files
    let realFiles = map (\fn -> lexiconDir ++ fn) $ validFiles

    maps <- sequence $ map (parseFile lexiconDir) realFiles
    let lexicon = toList $ unionMaps maps
    return lexicon

parseFile :: [Char] -> [Char] -> IO (Map ([Char], [Char], [Char]) Int)
parseFile basedir inputFileName = do
    lexicon <- readFile $ basedir ++ inputFileName
    let kvMap = fromList $ map (\l -> Prelude.read l :: (([Char], [Char], [Char]), Int)) $ lines lexicon
    return kvMap
