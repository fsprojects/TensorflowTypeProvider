namespace TensorflowTypeProvider

// TODO: MEMORY!!! The Tensorflow Protobuf Type Provider may consume a fair amount of memory
// BUG: rose tree query can end up with a key that is not in the graph cache.
// TODO: Create Op Specific Types. These could be code-genned from the Tensorflow Protobuf Defintions
// TODO: Improve XMLDoc
// TODO: Member which returns an array with the correct arity, e.g. float[,] for NPY/NPZ

open FSharp.Core.CompilerServices
open FSharp.Quotations
open NPYReaderWriter
open ProviderImplementation.ProvidedTypes
open System
open System.IO
open System.Numerics
open System.Reflection
open Tensorflow
open Microsoft.FSharp.Quotations
open Microsoft.FSharp.Quotations.Patterns

// NHWC
// offset_nhwc(n, c, h, w) = n * HWC + h * WC + w * C + c
module Array2D = 
    let ofArray<'a> (H:int) (W:int) (arr:Array) = 
        // C row major is assumed
        if arr.Rank = 1 then 
            if arr.Length = H * W then
                Array2D.init H W (fun h w -> arr.GetValue(h*W + w) :?> 'a)
            else
                failwithf "Shape missmatch. Expected count of %i found %i" (H*W) arr.Length
        elif arr.Rank = 2 then 
            if arr.GetLength(0) = H && arr.GetLength(1) = W then
                arr :?> 'a[,] 
            else failwithf "Shape missmatch. Expected (%i,%i) found (%i,%i)" H W (arr.GetLength(0)) (arr.GetLength(1))
        else failwithf "Rank 1 or 2 expected, rank %i found" arr.Rank

module Array3D = 
    let ofArray<'a> (H:int) (W:int) (C:int) (arr:Array) = 
        // C row major is assumed
        if arr.Rank = 1 then 
            if arr.Length = H * W * C then
                Array3D.init H W C (fun h w c-> arr.GetValue(h*W*C + w*C + c) :?> 'a)
            else
                failwithf "Shape missmatch. Expected count of %i found %i" (H*W*C) arr.Length
        elif arr.Rank = 3 then 
            if arr.GetLength(0) = H && arr.GetLength(1) = W && arr.GetLength(2) = C then
                arr :?> 'a[,,] 
            else failwithf "Shape missmatch. Expected (%i,%i,%i) found (%i,%i,%i)" H W C (arr.GetLength(0)) (arr.GetLength(1)) (arr.GetLength(2))
        else failwithf "Rank 1 or 3 expected, rank %i found" arr.Rank

module Array4D = 
    let ofArray<'a> (N:int) (H:int) (W:int) (C:int) (arr:Array) = 
        // C row major is assumed
        if arr.Rank = 1 then 
            if arr.Length = N * H * W * C then
                Array4D.init N H W C (fun n h w c -> arr.GetValue(n*H*W*C + h*W*C + w*C + c) :?> 'a)
            else
                failwithf "Shape missmatch. Expected count of %i found %i" (N*H*W*C) arr.Length
        elif arr.Rank = 4 then 
            if arr.GetLength(0) = N && arr.GetLength(1) = H && arr.GetLength(2) = W && arr.GetLength(3) = C then
                arr :?> 'a[,,,] 
            else failwithf "Shape missmatch. Expected (%i,%i,%i,%i) found (%i,%i,%i,%i)" 
                    N H W C (arr.GetLength(0)) (arr.GetLength(1)) (arr.GetLength(2)) (arr.GetLength(3))
        else failwithf "Rank 1 or 4 expected, rank %i found" arr.Rank

module String =
    let take n (xs:string) = if xs.Length > n then xs.Substring(0,n) else xs

[<AutoOpen>]
module Utils = 
    // TODO Wanted to use NDArray to do the conversions from Bytes to Data but the conversions 
    // I wanted did not seem to be supported
    let parseData(data:byte[],dt:DataType) : Array =
        let byteSize(dt:DataType) =
            match dt with
            | DataType.DtFloat -> 4
            | DataType.DtDouble ->8
            | DataType.DtUint8 -> 1
            | DataType.DtUint16 -> 2
            | DataType.DtUint32 -> 4
            | DataType.DtUint64 -> 8
            | DataType.DtInt8 -> 1
            | DataType.DtInt16 -> 2
            | DataType.DtInt32 -> 4
            | DataType.DtInt64 -> 8
            | DataType.DtString -> failwith "string should not have a byte size"
            | DataType.DtComplex128 -> 16
            | DataType.DtBool -> 1
            | _ -> failwith "unsuported"
        let bs = byteSize(dt)
        if data.Length % bs <> 0 then failwith "Data size is not a multiple of DataType ByteSize"
        let count = data.Length / bs;
        // TODO clean this up a bit
        match dt with
        | DataType.DtFloat ->
            [| for i in 0 .. count - 1 ->   BitConverter.ToSingle(data,i*bs)|] :> Array
        | DataType.DtDouble -> 
            [| for i in 0 .. count - 1 ->   BitConverter.ToDouble(data,i*bs)|] :> Array
        | DataType.DtUint8 -> 
            data |> Array.map uint8 :> Array
        | DataType.DtUint16 -> 
            [| for i in 0 .. count - 1 ->   BitConverter.ToUInt16(data,i*bs)|] :> Array
        | DataType.DtUint32 -> 
            [| for i in 0 .. count - 1 ->   BitConverter.ToUInt32(data,i*bs)|] :> Array
        | DataType.DtUint64 -> 
            [| for i in 0 .. count - 1 ->   BitConverter.ToUInt64(data,i*bs)|] :> Array
        | DataType.DtInt8 -> 
            data |> Array.map int8 :> Array
        | DataType.DtInt16 -> 
            [| for i in 0 .. count - 1 ->   BitConverter.ToInt16(data,i*bs)|] :> Array
        | DataType.DtInt32 -> 
            [| for i in 0 .. count - 1 ->   BitConverter.ToInt32(data,i*bs)|] :> Array
        | DataType.DtInt64 -> 
            [| for i in 0 .. count - 1 ->   BitConverter.ToInt64(data,i*bs)|] :> Array
        | DataType.DtComplex128 -> 
            failwith "todo"
        | DataType.DtBool -> 
            data |> Array.map ((<>) 0uy) :> Array
        | _ -> failwith "unsuported"
    // replaces chars which require backticks
    // This function is not perfect but good enough for now
    // Note this is not an invertable mapping so be careful to handle duplicates
    let replaceBacktickChars =
        let prohibitedChars = Set(""" .,+$&[]/\*", `@""".ToCharArray())
        fun (name:string) -> 
            if String.IsNullOrWhiteSpace(name) then "_" 
            else 
                name |> String.map (fun c -> if prohibitedChars.Contains(c) then '_' else c)
                |> fun name -> if Char.IsLetter(name.[0]) then name else sprintf "_%s" name

    let getMakeUnique() =
        let mutable m = Map.empty<string,int>
        fun s -> 
            let count = match m.TryFind(s) with | None -> 0 | Some(count) -> count
            m <- m.Add(s,count+1)
            if count = 0 then s
            else sprintf "%s_%i" s count |> fun s' -> m <- m.Add(s',1); s'

    //let filterRoot (xs:string[]) = xs |> Array.filter (fun x -> x.Contains(sprintf "%c" '/'))

    let expand (keys:string[]) (map:Map<string,string[]>) (prefix:string) =
        keys
        |> Array.filter (fun x -> x.StartsWith(prefix)) 
        |> Array.collect (fun x -> 
            match map.TryFind(x) with 
            | Some(xs) -> xs 
            | _ -> failwith "the provided map should contain all keys")
        |> Array.filter (fun x -> not(x.StartsWith(prefix)))
        //|> filterRoot // NOTE: It's often that the Consts are root

    let findRelative (xs:string[]) (ys:string[]) =
        let zs = 
            if ys.Length < xs.Length then ys
            else
                let common = 
                    (xs,ys |> Array.take (xs.Length)) 
                    ||> Array.zip |> Array.takeWhile (fun (x,y) -> x=y) |> Array.map fst
                if common.Length < ys.Length 
                then [| yield! common; yield ys.[common.Length]|]
                else common // This shouldn't happen
        (zs,zs.Length - xs.Length)



module Map =
    let invert (x:Map<'a,'b[]>) = 
        [| for KeyValue(k,vs) in x do for v in vs do yield (v,k) |]
        |> Array.groupBy fst 
        |> Array.map (fun (x,xs) -> (x,xs |> Array.map snd))
        |> Map
                
type RoseTree<'a> =  | Node of 'a*RoseTree<'a>[]

type TreeNames = { full : string; local : string; display : string }

module RoseTree = 

    let rec map (f:'a->'b) (tree:RoseTree<'a>) =
        match tree with
        | Node(x,[||]) -> Node(f(x),[||])
        | Node(x,xs) -> Node(f(x),xs |> Array.map (map f))

    let rec group (xss:string list list) : RoseTree<string>[] =
        match xss with 
        | [] -> [||] 
        | [[]] -> [||]
        | _ -> 
            xss
            |> List.groupBy (function | [] -> "" | h::_ -> h)
            |> List.map (fun (k,xs) -> Node(k,group (xs |> List.map (function | [] -> [] | _::tail -> tail) |> List.distinct)))
            |> List.toArray

    let rec mapChildren (f:'a[] -> 'a[]) (rt:RoseTree<'a>) : RoseTree<'a> =
        match rt with
        | Node(x,[||]) -> Node(x,[||])
        | Node(x,xs) -> 
            let xs,yss = xs |> Array.map (function | Node(x,ys) -> (x,ys)) |> Array.unzip
            let childNodes = yss |> Array.map (fun ys -> ys |> Array.map (mapChildren f))
            // Possible array problem with different sizes??
            Node(x,(f(xs),childNodes) ||> Array.zip |> Array.map (fun (x,ys) -> Node(x,ys)))
            

    let distinctDisplayNames(xs:TreeNames[]) : TreeNames[] =
        xs |> Array.scan (fun (_,acc:Map<string,int>) x -> 
            match acc.TryFind(x.display) with
            | None -> (x,acc.Add(x.display,1)) 
            | Some(count) -> 
                let z' = sprintf "%s_%i" x.display count
                ({x with display = z'},acc.Add(z',0).Add(x.display,count+1))
            )  ({full = "";local = "";display = ""},Map.empty<string,int>) 
            |> Array.skip 1
            |> Array.map fst

    // These functions are a bit weird
    // This weirdness is only for when distinct names map to the same display name
    let getDistinctNames (separator:char) (names:string[]) : RoseTree<TreeNames>[] = 
        // Note the function f must not change the length of the array
        let rec mapAcc (f:'a list ->'b) (acc:'a list) (tree:RoseTree<'a>) =
            match tree with
            | Node(x,[||]) -> Node(f(x::acc),[||])
            | Node(x,xs) -> Node(f(x::acc),xs |> Array.map (mapAcc f (x::acc)))
        names 
        |> Array.map (fun x -> x.Split(separator) |> List.ofArray) 
        |> Array.toList |> group
        |> Array.map (
            mapAcc (
                function | [] -> failwith "never - given the group function" 
                         | (head::tail) -> 
                            { full = (head::tail) |> List.rev |> String.concat "/";
                              display = head;
                              local = Path.GetFileNameWithoutExtension(head) |> replaceBacktickChars}) [])
        |> Array.map (mapChildren distinctDisplayNames)
        
    let print (f:'a->string) (roseTrees:RoseTree<'a>[]) =
        let rec printRoseTree (depth:int) (roseTrees:RoseTree<'a>[]) =
            for roseTree in roseTrees do
                match roseTree with
                | Node(x,xs) ->
                    printfn "%s %s" ("".PadLeft(depth * 2)) (f(x))
                    printRoseTree (depth + 1) xs
        printRoseTree 0 roseTrees

    let rec queryRoseTrees (query:string list) (f:'a->string) (roseTrees:RoseTree<'a>[]) : RoseTree<'a> option =
        match query with
        | [] -> None
        | (head::tail) -> 
            roseTrees 
            |> Array.tryFind (function Node(x,_) -> f(x) = head) 
            |> Option.bind (
                function Node(x,xs) as n-> match tail with | [] -> Some(n) | _ -> queryRoseTrees tail f xs)

[<AutoOpen>]
module TF =
        
    type DataType with
        member this.Name =
            match this with
            | DataType.DtHalf -> "float16"
            | DataType.DtFloat -> "float32"
            | DataType.DtDouble -> "float64"
            | DataType.DtUint8 -> "uint8"
            | DataType.DtUint16 -> "uint16"
            | DataType.DtUint32 -> "uint32"
            | DataType.DtUint64 -> "uint64"
            | DataType.DtInt8 -> "int8"
            | DataType.DtInt16 -> "int16"
            | DataType.DtInt32 -> "int32"
            | DataType.DtInt64 -> "int64"
            | DataType.DtString -> "string"
            | DataType.DtComplex64 -> "complex64"
            | DataType.DtComplex128 -> "complex128"
            | DataType.DtBool -> "bool"
            | DataType.DtBfloat16 -> "bfloat16"
            | DataType.DtResource -> "resource"
            | DataType.DtVariant -> "variant"
            | DataType.DtHalfRef -> "float16 ref"
            | DataType.DtFloatRef -> "float32 ref"
            | DataType.DtDoubleRef -> "float64 ref"
            | DataType.DtUint8Ref -> "uint8 ref"
            | DataType.DtUint16Ref -> "uint16 ref"
            | DataType.DtUint32Ref -> "uint32 ref"
            | DataType.DtUint64Ref -> "uint64 ref"
            | DataType.DtInt8Ref -> "int8 ref"
            | DataType.DtInt16Ref -> "int16 ref"
            | DataType.DtInt32Ref -> "int32 ref"
            | DataType.DtInt64Ref -> "int64 ref"
            | DataType.DtStringRef -> "string ref"
            | DataType.DtComplex64Ref -> "complex64 ref"
            | DataType.DtComplex128Ref -> "complex128 ref"
            | DataType.DtBoolRef -> "bool ref"
            | DataType.DtBfloat16Ref -> "bfloat16 ref"
            | DataType.DtResourceRef -> "resource ref"
            | DataType.DtVariantRef -> "variant ref"
            | _ -> String.Empty
        
        member this.ToType =
            match this with
            | DataType.DtFloat -> typedefof<single>
            | DataType.DtDouble -> typedefof<double>
            | DataType.DtUint8 -> typedefof<uint8>
            | DataType.DtUint16 -> typedefof<uint16>
            | DataType.DtUint32 -> typedefof<uint32>
            | DataType.DtUint64 -> typedefof<uint64>
            | DataType.DtInt8 -> typedefof<int8>
            | DataType.DtInt16 -> typedefof<int16>
            | DataType.DtInt32 -> typedefof<int32>
            | DataType.DtInt64 -> typedefof<int64>
            | DataType.DtString -> typedefof<string> // TFString?
            | DataType.DtComplex128 -> typedefof<Complex>
            | DataType.DtBool -> typedefof<bool>
            | DataType.DtFloatRef -> typedefof<single>
            | DataType.DtDoubleRef -> typedefof<double>
            | DataType.DtUint8Ref -> typedefof<uint8>
            | DataType.DtUint16Ref -> typedefof<uint16>
            | DataType.DtUint32Ref -> typedefof<uint32>
            | DataType.DtUint64Ref -> typedefof<uint64>
            | DataType.DtInt8Ref -> typedefof<int8>
            | DataType.DtInt16Ref -> typedefof<int16>
            | DataType.DtInt32Ref -> typedefof<int32>
            | DataType.DtInt64Ref -> typedefof<int64>
            | DataType.DtStringRef -> typedefof<string> // TFString?
            | DataType.DtComplex128Ref -> typedefof<Complex>
            | DataType.DtBoolRef -> typedefof<bool>
            | _ -> typedefof<obj>

    type TensorShapeProto with
        member this.FriendlyName =
            if this.UnknownRank then "unknown"
            // TODO Consider how to handle named dimensions.
            else [| for x in this.Dim -> if x.Size < 0L then "?" else sprintf "%i" x.Size|]
                 |> String.concat "," |> sprintf "(%s)"

type INPYTensor =
    abstract member dtype : NPYDType with get
    abstract member shape : int[] with get
    abstract member data : Array with get
    abstract member foretran_order : bool with get
    abstract member path : string option with get
    abstract member name : string option with get

type NPYTensor(path:string) =
    let npy = lazy (NPYReaderWriter.readNumpy(File.ReadAllBytes(path)) |> Result.requireOk)
    interface INPYTensor with
        member this.dtype with get() : NPYDType = npy.Force() |> fst |> fun x -> x.npyDType
        member this.shape with get() : int[] = npy.Force() |> fst |> fun x -> x.shape
        member this.data with get() : Array = npy.Force() |> snd
        member this.foretran_order with get() : bool = npy.Force() |> fst |> fun x -> x.fortran_order
        member this.path with get() = Some(path)
        member this.name with get() = None


type NPZTensor(desc:NPYDescription,array:Array,path:string,name:string) =
    interface INPYTensor with
        member this.dtype with get() : NPYDType = desc.npyDType
        member this.shape with get() : int[] = desc.shape
        member this.data with get() : Array = array
        member this.foretran_order with get() : bool = desc.fortran_order
        member this.path with get() = Some(path)
        member this.name with get() = Some(name)


type SomeRuntimeHelper() = 
    static member Help() = "help"

type TFGraphCache(path:string,separator:char) = 
    let data = File.ReadAllBytes(path)
    let mutable graph = Some(GraphDef.Parser.ParseFrom(data))
    let indexMap = Map(graph.Value.Node |> Seq.mapi (fun i x -> (x.Name,i)))
    let keys = graph.Value.Node |> Seq.map (fun x -> x.Name) |> Seq.toArray
    let inputMap = Map(graph.Value.Node |> Seq.map (fun x -> (x.Name,x.Input |> Seq.toArray)))
    let outputMap = inputMap |> Map.invert
    let getNodeByName(name:string) = graph.Value.Node |> Seq.find (fun x -> x.Name = name)
    let ops = 
        graph.Value.Node 
        |> Seq.map (fun x -> (x.Op,x.Name)) 
        |> Seq.toArray 
        |> Array.groupBy fst
        |> Array.map (fun (x,xs) -> (x,xs |>Array.map snd))
        |> Map
    let names = graph.Value.Node |> Seq.map (fun x -> x.Name) |> Seq.toArray
    let roseTrees = names |> RoseTree.getDistinctNames separator
    let splitPath (x:string) = x.Split([|separator|],StringSplitOptions.RemoveEmptyEntries)
    let expandRoseTree (path:string) (map:Map<string,string[]>) : (int*RoseTree<TreeNames>*string[])[]=
        let xs = path |> splitPath
        [|
            // Dedupe
            let queryDepthPairs = 
                [|for target in expand keys map path -> findRelative xs (target |> splitPath)|]
                |> Array.groupBy fst |> Array.map (fun (query,xs) -> (query,snd xs.[0]))
            for (query,relDepth) in queryDepthPairs do
                // TODO could clean this up a bit
                match RoseTree.queryRoseTrees (query |> Array.toList) (fun (x:TreeNames) -> x.local) roseTrees with
                | Some(rt) -> yield (relDepth,rt,query)
                | _ -> ()
        |]
    let attrs = [|for x in graph.Value.Node -> 
                    [|for KeyValue(k,v) in x.Attr do 
                        if v.ValueCase <> AttrValue.ValueOneofCase.Tensor then yield (k,v)|]|]

    let tensors = [|for x in graph.Value.Node -> 
                    [|for KeyValue(k,v) in x.Attr do 
                        if v.ValueCase = AttrValue.ValueOneofCase.Tensor then yield (k,(v.Tensor.Dtype,v.Tensor.TensorShape))|]|]
    do graph <- None // an attempt to free up memory
    member this.GetAttr(index:int) = attrs.[index]
    member this.GetTensors(index:int) = tensors.[index]
    member this.IndexMap = indexMap
    member this.InputMap = inputMap
    member this.OutputMap = outputMap
    member this.GetNodeByName(name) = getNodeByName(name)
    member this.Keys = keys
    member this.Ops = ops
    member this.Names = names
    member this.RoseTrees = roseTrees
    member this.SplitPath(path) = splitPath(path)
    member this.ExpandRoseTree (path:string) (map:Map<string,string[]>)  = expandRoseTree path map 

type Cache<'k,'v when 'k : comparison>() =
    let mutable cache = Map.empty<'k,'v>
    member this.TryFetch(key:'k,f:unit->'v) : 'v = 
        cache.TryFind(key) |> Option.defaultWith (fun () -> f() |> fun v -> cache <- cache.Add(key,v); v )
    member this.Clear() = cache <- Map.empty

[<TypeProvider>]
type TFProvider (config : TypeProviderConfig) as this =
    inherit TypeProviderForNamespaces (config, addDefaultProbingLocation=true)
    let ns = "TensorflowTypeProvider"
    let asm = Assembly.GetExecutingAssembly()
    let makeUnique = getMakeUnique()
    static let mutable tfGraphCache = Cache<string*char,TFGraphCache>()
    static let mutable npyCache = Cache<string,NPYDescription>()
    static let mutable npzCache = Cache<string,Map<string,NPYDescription>>()

    // check we contain a copy of runtime files, and are not referencing the runtime DLL
    do assert (typeof<SomeRuntimeHelper>.Assembly.GetName().Name = asm.GetName().Name)  

    let rec getMethodInfo expr =
        match expr with
        | Call(_,mi,_) -> Some(mi)
        | Lambda(_,expr) -> getMethodInfo expr
        | _ -> None

    // NOTE: Should be possible to have a general method which takes in a shape of int[] and returns the correctly shaped array
    let mis = 
        [| <@@ Array2D.ofArray @@>; <@@ Array3D.ofArray @@>; <@@ Array4D.ofArray @@> |]
        |> Array.map (fun x -> (x |> getMethodInfo |> Option.get).DeclaringType.GetMember("ofArray").[0] :?> MethodInfo)

    let addValues(myType:ProvidedTypeDefinition,desc) =
        let rank = desc.shape.Length
        match desc.npyDType.TryToType() with
        | None -> ()
        | Some(t) -> 
            let getTypedData(args:Expr list) =
                match rank with
                | 2 | 3 | 4-> 
                    let shape = desc.shape;
                    Expr.Call(mis.[rank-2].MakeGenericMethod(t),
                        [yield! [for r in 0..rank - 1 -> Expr.Value(shape.[r])];  
                         yield <@@ (%%args.[0]:INPYTensor).data @@>])
                | 1 -> <@@ (%%args.[0]:INPYTensor).data @@>
                | _ -> failwithf "usupported rank %i" rank
            let pp = ProvidedProperty("Values",t.MakeArrayType(rank),getterCode = getTypedData)
            pp.AddXmlDoc(sprintf "<summary>%O</summary>" desc)
            myType.AddMember(pp)

    let t = 
        let t = ProvidedTypeDefinition(asm, ns, "TFProvider", Some typeof<obj>, isErased=true)
        t.DefineStaticParameters( [
            ProvidedStaticParameter("Path", typeof<string>);
            ProvidedStaticParameter("Separator", typeof<char>,'/')
        ], (fun typeName args -> 
            match args with
            | [|:? string as path;:? char as separator |] ->
            if not(File.Exists(path)) then failwithf "File %s not found" path
            else
                match System.IO.Path.GetExtension(path).ToLower() with
                | ".npy" -> 
                    let myType = ProvidedTypeDefinition(asm, ns, typeName, Some typeof<INPYTensor>, isErased=true)
                    myType.AddMember(ProvidedConstructor([], invokeCode = (fun _ -> <@@ NPYTensor(path) :> INPYTensor @@>)))
                    let getDesc path = File.ReadAllBytes(path) |> NPYReaderWriter.readNumpy |> Result.requireOk |> fst 
                    let desc = npyCache.TryFetch(path,fun () -> getDesc path)
                    addValues(myType, desc)
                    myType.AddXmlDoc(sprintf "<summary>%O</summary>" desc)
                    myType
                | ".npz" -> 
                    let myType = ProvidedTypeDefinition(asm, ns, typeName, Some typeof<Map<string,INPYTensor>>, isErased=true)
                    let expr = <@@ 
                                    File.ReadAllBytes(path)
                                    |> NPYReaderWriter.readFromNPZ
                                    |> Map.map (fun k (desc,arr) -> NPZTensor(desc,arr,path,k) :> INPYTensor)
                                @@>
                    let ctor = ProvidedConstructor([], invokeCode = (fun args -> expr))
                    myType.AddMember(ctor)
                    let getDescMap path = 
                        File.ReadAllBytes(path)
                        |> NPYReaderWriter.readFromNPZ
                        |> Map.map (fun k (v,_) -> v)
                    let descMap = npzCache.TryFetch(path,fun () -> getDescMap path)
                    let names = [|for KeyValue(k,v) in descMap -> k|]
                    let roseTrees = names |> RoseTree.getDistinctNames separator
                    let rec procNodes (myType:ProvidedTypeDefinition) (nodes:RoseTree<TreeNames>[]) =
                        for node in nodes do
                            match node with
                            | Node(x,[||]) ->
                                let xFull = x.full
                                let xDisplay = Path.GetFileNameWithoutExtension(x.display) // NOTE: Shouldn't this be done earlier?
                                let subType = ProvidedTypeDefinition(asm, ns, xDisplay,Some typeof<INPYTensor>,isErased=true)
                                let pp = ProvidedProperty(xDisplay,subType, getterCode = 
                                            (fun args -> <@@ (%%args.[0]:Map<string,INPYTensor>).Item(xFull) @@>))
                                pp.AddXmlDoc(sprintf "<summary>%s %O</summary>" x.local descMap.[x.full])
                                addValues(subType,descMap.[x.full])
                                myType.AddMembers([subType :> MemberInfo;pp :> MemberInfo])
                            | Node(x,xs) ->
                                let xFull = x.full
                                let subType = ProvidedTypeDefinition(asm, ns, x.display,Some typeof<Map<string,INPYTensor>>,isErased=true)
                                procNodes subType xs
                                myType.AddMember(subType)
                                let f (args:Expr list) = 
                                    <@@ 
                                        (%%args.[0]:Map<string,INPYTensor>)
                                        |> Map.filter (fun k _ -> k.StartsWith(xFull))
                                     @@>
                                // TODO consider adding a property to get the Map w/o the prefixs, could be useful..
                                let pp = ProvidedProperty(x.display,subType,getterCode = f)
                                pp.AddXmlDoc(sprintf "<summary>%s</summary>" x.full)
                                myType.AddMember(pp)
                    procNodes myType roseTrees
                    myType
                | ".pb" ->
                    let cache = tfGraphCache.TryFetch((path,separator), fun () -> TFGraphCache(path,separator));
                    let myType = ProvidedTypeDefinition(asm, ns, typeName, Some typeof<GraphDef>, isErased=true)
                    myType.AddMember(ProvidedConstructor([], invokeCode = (fun _ -> <@@ GraphDef.Parser.ParseFrom(File.ReadAllBytes(path)) @@>)))

                    let recPropGraph(name) = 
                        let ptd = ProvidedTypeDefinition(asm,ns,makeUnique(name), Some typeof<obj>, isErased=true,hideObjectMethods=true)
                        let pp = ProvidedProperty(name,ptd,getterCode = (fun args -> <@@ box(%%args.[0]) @@>))
                        (ptd,pp)

                    let recPropGraphDefToObj(name) = 
                        let ptd = ProvidedTypeDefinition(asm,ns,makeUnique(name), Some typeof<obj>, isErased=true,hideObjectMethods=true)
                        let pp = ProvidedProperty(name,ptd,getterCode = (fun args -> <@@ box(%%args.[0]:GraphDef) @@>))
                        (ptd,pp)

                    let recPropNodeDefToObj(name) = 
                        let ptd = ProvidedTypeDefinition(asm,ns,makeUnique(name), Some typeof<obj>, isErased=true,hideObjectMethods=true)
                        let pp = ProvidedProperty(name,ptd,getterCode = (fun args -> <@@ box(%%args.[0]:NodeDef) @@>))
                        (ptd,pp)

                    let rec nodeFromGraph(displayName:string, fullName:string) = 
                        let index = cache.IndexMap.[fullName]
                        let ptd = ProvidedTypeDefinition(asm,ns,makeUnique(displayName), Some typeof<NodeDef>, isErased=true,hideObjectMethods=true)
                        let pp = ProvidedProperty(displayName,ptd,getterCode = (fun args -> <@@ ((%%args.[0]:obj) :?> GraphDef).Node.[index] @@>))
                        // TODO Lift the AttrList onto the above NodeDef
                        ptd.AddMembersDelayed(fun () ->
                            let ptd = ProvidedTypeDefinition(asm,ns,"AttrList", Some typeof<obj>, isErased=true,hideObjectMethods=true)
                            let pp = ProvidedProperty("AttrList",ptd,getterCode = (fun args -> <@@ box(%%args.[0]:NodeDef) @@>))
                            ptd.AddMembersDelayed(fun () -> [
                                for (k,v) in cache.GetAttr(index) do
                                    match v.ValueCase with
                                    | AttrValue.ValueOneofCase.B ->     
                                        yield ProvidedProperty(k,typeof<bool>,getterCode = 
                                            fun args -> <@@ ((%%args.[0]:obj) :?> NodeDef).Attr.[k].B @@>) 
                                        |> fun x -> x.AddXmlDoc(sprintf "bool: %b" v.B); x :> MemberInfo
                                    | AttrValue.ValueOneofCase.F -> 
                                        yield ProvidedProperty(k,typeof<float32>,getterCode = 
                                            fun args -> <@@ ((%%args.[0]:obj) :?> NodeDef).Attr.[k].F @@>) 
                                        |> fun x -> x.AddXmlDoc(sprintf "float32: %f" v.F); x :> MemberInfo
                                    | AttrValue.ValueOneofCase.Func -> 
                                        //v.Func.Name
                                        //v.Func.Attr
                                        () // TODO, this will require a new sub type and recurssion
                                    | AttrValue.ValueOneofCase.List -> 
                                        if v.List.Type.Count > 0 then 
                                            yield ProvidedProperty(k,typeof<DataType[]>,getterCode = 
                                                fun args -> <@@ ((%%args.[0]:obj) :?> NodeDef).Attr.[k].List.Type |> Seq.toList @@>) :> MemberInfo
                                        elif v.List.B.Count > 0 then 
                                            yield ProvidedProperty(k,typeof<bool[]>,getterCode = 
                                                fun args -> <@@ ((%%args.[0]:obj) :?> NodeDef).Attr.[k].List.B |> Seq.toList @@>) :> MemberInfo
                                        elif v.List.F.Count > 0 then 
                                            yield ProvidedProperty(k,typeof<float32[]>,getterCode = 
                                                fun args -> <@@ ((%%args.[0]:obj) :?> NodeDef).Attr.[k].List.F |> Seq.toList @@>) :> MemberInfo
                                        elif v.List.S.Count > 0 then 
                                            yield ProvidedProperty(k,typeof<string[]>,getterCode = 
                                                fun args -> <@@ ((%%args.[0]:obj) :?> NodeDef).Attr.[k].List.S |> Seq.toList @@>) :> MemberInfo
                                        elif v.List.Shape.Count > 0 then 
                                            yield ProvidedProperty(k,typeof<TensorShapeProto[]>,getterCode = 
                                                fun args -> <@@ ((%%args.[0]:obj) :?> NodeDef).Attr.[k].List.Shape |> Seq.toList @@>) :> MemberInfo
                                        elif v.List.I.Count > 0 then 
                                            yield ProvidedProperty(k,typeof<int64[]>,getterCode = 
                                                fun args -> <@@ ((%%args.[0]:obj) :?> NodeDef).Attr.[k].List.I |> Seq.toList @@>) :> MemberInfo
                                        // TODO List of Tensors as Values Array[], not able to mix different shapes
                                        elif v.List.Tensor.Count > 0 then 
                                            yield ProvidedProperty(k,typeof<TensorProto>,getterCode = 
                                                fun args -> <@@ ((%%args.[0]:obj) :?> NodeDef).Attr.[k].List.Tensor |> Seq.toList @@>) :> MemberInfo
                                        elif v.List.Func.Count > 0 then ()
                                        else () // ignore
                                    | AttrValue.ValueOneofCase.Placeholder -> 
                                        yield ProvidedProperty(k,typeof<string>,getterCode = 
                                            fun args -> <@@ ((%%args.[0]:obj) :?> NodeDef).Attr.[k].Placeholder @@>) 
                                        |> fun x -> x.AddXmlDoc(sprintf "Placeholder: %s" v.Placeholder); x :> MemberInfo
                                    | AttrValue.ValueOneofCase.I -> 
                                        yield ProvidedProperty(k,typeof<int64>,getterCode = 
                                            fun args -> <@@ ((%%args.[0]:obj) :?> NodeDef).Attr.[k].I@@>) 
                                        |> fun x -> x.AddXmlDoc(sprintf "int64: %i" v.I); x :> MemberInfo
                                    | AttrValue.ValueOneofCase.None -> () 
                                    | AttrValue.ValueOneofCase.S -> 
                                        yield ProvidedProperty(k,typeof<Google.Protobuf.ByteString>,getterCode = 
                                            fun args -> <@@ ((%%args.[0]:obj) :?> NodeDef).Attr.[k].S@@>) 
                                        |> fun x -> x.AddXmlDoc(sprintf "ByteString: %s" (v.S.ToStringUtf8() |> String.take 200)); x :> MemberInfo
                                    | AttrValue.ValueOneofCase.Shape -> 
                                        yield ProvidedProperty(k,typeof<TensorShapeProto>,getterCode = 
                                            fun args -> <@@ ((%%args.[0]:obj) :?> NodeDef).Attr.[k].Shape@@>) 
                                        |> fun x -> x.AddXmlDoc(sprintf "Shape: %s" (v.Shape.FriendlyName)); x :> MemberInfo
                                    | AttrValue.ValueOneofCase.Tensor -> 
                                        failwith "This should not happen as it is filtered out at an earlier stage"
                                    | AttrValue.ValueOneofCase.Type -> 
                                        yield ProvidedProperty(k,typeof<DataType>,getterCode =
                                            fun args -> <@@ ((%%args.[0]:obj) :?> NodeDef).Attr.[k].Type@@>) 
                                        |> fun x -> x.AddXmlDoc(sprintf "DataType: %s" (v.Type.Name)); x :> MemberInfo
                                    | _ -> () //failwith "error"
                                for (k,(dt,shape)) in cache.GetTensors(index) do
                                        let ptd = ProvidedTypeDefinition(asm,ns,"Tensor", Some typeof<TensorProto>,isErased=true)
                                        yield ptd :> MemberInfo
                                        let docString = sprintf "Tensor: %s %s" (dt.Name) (shape.FriendlyName)
                                        yield ProvidedProperty(k,ptd,getterCode =
                                            fun args -> <@@ ((%%args.[0]:obj) :?> NodeDef).Attr.[k].Tensor@@>) 
                                        |> fun x -> x.AddXmlDoc(docString); x :> MemberInfo
                                        // Add Extra "Values" member if we can
                                        if not(shape.UnknownRank) && not(shape.Dim |> Seq.exists (fun d -> d.Size < 0L)) then
                                            let dims = shape.Dim |> Seq.toArray |> Array.map (fun d -> int d.Size)
                                            //let size = dims |> Array.fold (*) 1
                                            let rank = dims.Length
                                            let t = dt.ToType 
                                            if t <> typeof<obj> then
                                                let getTypedData(args:Expr list) =
                                                    let arrExpr = <@@ parseData((%%args.[0]:TensorProto).TensorContent.ToByteArray(),dt) @@>
                                                    match rank with
                                                    | 2 | 3 | 4-> 
                                                        Expr.Call(mis.[rank-2].MakeGenericMethod(t),
                                                            [yield! [for r in 0..rank - 1 -> Expr.Value(dims.[r])];  
                                                             yield arrExpr])
                                                    | 1 -> arrExpr 
                                                    | _ -> failwithf "usupported rank %i" rank
                                                let pp = ProvidedProperty("Values",t.MakeArrayType(rank),getterCode = getTypedData)
                                                pp.AddXmlDoc(docString)
                                                ptd.AddMember(pp)
                                ])

                            // TODO If we want to return to the Graph we must keep a reference to the graph which means 
                            //  proxying the NodeDef type
//                            let f (name:string) (map:Map<string,string[]>) : MemberInfo[] =
//                                let (subTypeDef,subPropDef) = recPropGraph(name)
//                                subTypeDef.AddMembersDelayed(fun () -> [
//                                    //let full = query |> String.concat (sprintf "%c" separator)
//                                    let full = fullName
//                                    for (depth,rt,query) in  cache.ExpandRoseTree full map do
//                                        let depthPrefix = if depth = 0 then "" else  "../" |> String.replicate -depth
//                                        match rt with
//                                        | RoseTree.Node(x,[||]) -> 
//                                            yield! nodeFromGraph(depthPrefix + x.display,x.full)
//                                        | RoseTree.Node(x,xs) -> 
//                                            yield! (threadGraphDefThroughRoseTree false query (depthPrefix + x.display) xs)
//                                ])
//                                [|subTypeDef :> MemberInfo; subPropDef :> MemberInfo|]
                            [
                                yield ptd :> MemberInfo
                                yield upcast pp
//                                yield! f "Inputs" cache.InputMap
//                                yield! f "Outputs" cache.OutputMap
                            ])
                        [|ptd :> MemberInfo;upcast pp|]
                    and threadGraphDefThroughRoseTree (isFirst:bool) (query:string[]) (name:string) (rts: RoseTree<TreeNames>[]) =
                        let (ptd,pp) = 
                            if isFirst then recPropGraphDefToObj(name) else recPropGraph(name)
                        ptd.AddMembersDelayed(fun () -> [
                            for rt in rts do
                                match rt with
                                | RoseTree.Node(x,[||]) -> 
                                    yield! nodeFromGraph(x.display,x.full) 
                                | RoseTree.Node(x,xs) -> 
                                    yield! (threadGraphDefThroughRoseTree false [|yield! query; yield x.local|] x.display xs) 
                            let f (name:string) (map:Map<string,string[]>) : MemberInfo[] =
                                let (subTypeDef,subPropDef) = recPropGraph(name)
                                subTypeDef.AddMembersDelayed(fun () -> [
                                    let full = query |> String.concat (sprintf "%c" separator)
                                    for (depth,rt,query) in  cache.ExpandRoseTree full map do
                                        let depthPrefix = if depth = 0 then "" else  "../" |> String.replicate -depth
                                        match rt with
                                        | RoseTree.Node(x,[||]) -> 
                                            yield! nodeFromGraph(depthPrefix + x.display,x.full)
                                        | RoseTree.Node(x,xs) -> 
                                            yield! (threadGraphDefThroughRoseTree false query (depthPrefix + x.display) xs)
                                ])
                                [|subTypeDef :> MemberInfo; subPropDef :> MemberInfo|]
                            yield! f "Inputs" cache.InputMap
                            yield! f "Outputs" cache.OutputMap
                        ])
                        [|ptd :> MemberInfo;pp :> MemberInfo|]
                    myType.AddMembers(
                        let (allNodesType,allNodesProperty) = recPropGraphDefToObj("AllNodes")
                        allNodesType.AddMembers([for name in cache.Names do yield! nodeFromGraph(name,name)])
                        let (allOpsType,allOpsProperty) = recPropGraphDefToObj("AllOps")
                        allOpsType.AddMembers([
                            for KeyValue(op,names) in cache.Ops do
                                let (opType,opProperty) = recPropGraph(op)
                                yield opType :> MemberInfo
                                yield opProperty :> MemberInfo
                                opType.AddMembers([ for name in names do yield! nodeFromGraph(name,name)])
                        ])
                        [
                            yield allNodesType :> MemberInfo
                            yield upcast allNodesProperty
                            yield upcast allOpsType
                            yield upcast allOpsProperty
                            yield! threadGraphDefThroughRoseTree true [||] "RootNodes" cache.RoseTrees
                        ]
                        )
                    myType
                | ".pbtxt" -> failwith "todo unsuported at this time"
                | ext -> failwithf "Unsupported file extension %s" ext
            | _ -> failwith "Unsupported arguments"
            ))
        t
    do this.AddNamespace(ns, [t])

[<assembly:CompilerServices.TypeProviderAssembly()>]
do ()

