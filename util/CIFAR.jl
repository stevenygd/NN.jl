module CIFAR

# package code goes here
function labelnames()
  # Set File Name
  DataDir = joinpath(dirname(@__FILE__),"..","data","bin")
  LabelNameFile = joinpath(DataDir,"batches.meta.txt")

  # Settings
  Labels = 10

  # Open file
  h_LabelNameFile = open(LabelNameFile)

  # Read Labels
  LabelNames = readlines(h_LabelNameFile)

  # Close File
  close(h_LabelNameFile)

  # Strip newlines
  for i=1:length(LabelNames)
    LabelNames[i] = LabelNames[i][1:end-1]
  end
  LabelNames=LabelNames[1:end-1]
  
  return LabelNames
end

function normalize(X)
    X = convert(Array{Float64},X)

    samples = size(X,2)

    for i=1:samples
      x = X[:,i]
      minx = minimum(x)
      maxx = maximum(x)
      ranx = maxx-minx

      X[:,i] = (x - minx) ./ ranx
    end

    return X
end

function toLuminance(dataset)
  features = size(dataset,1)
  pixels = convert(Int,features/3)
  # Get Color Channels
  r = dataset[1:pixels,:]
  g = dataset[pixels+1:2*pixels,:]
  b = dataset[2*pixels+1:end,:]
  # Convert to floating point
  r = convert(Array{Float64},r)
  g = convert(Array{Float64},g)
  b = convert(Array{Float64},b)
  # Apply Luminance Projection
  lum = 0.2126*r .+ 0.7152*g .+ 0.0722*b;

  return lum
end

function traindata(;batch_number=-1, normalize_images=false, grey=false)
  Pixels = 1024
  Labels = 10

  if batch_number > 0
    # Assert to the number of batches  
    @assert minimum(batch_number) >= 1 && maximum(batch_number) <= 5

    # Set File Name
    DataDir = joinpath(dirname(@__FILE__),"..","data","bin")
    BatchFile = @sprintf "data_batch_%d.bin" batch_number
    BatchFile = joinpath(DataDir,BatchFile)
    LabelNameFile = joinpath(DataDir,"batches.meta.txt")
    
    # Fixed dataset values
    Features = 3*Pixels
    Samples = 10000    

    # Initialize Dataset Structures
    Dataset = Array(Float64,Features,Samples)
    Labels = Array(UInt8,Samples,1)
    LabelNames = Array(AbstractString,10,1)

    # Open Batch File
    h_BatchFile = open(BatchFile)

    # Loop Over Samples
    r = Array(UInt8,Pixels,1)
    g = Array(UInt8,Pixels,1)
    b = Array(UInt8,Pixels,1)
    for sampleIdx = 1:Samples
      # Read Label Byte
      Labels[sampleIdx] = read(h_BatchFile,UInt8)+1
      # Read Color Byte Arrays
      read!(h_BatchFile,r)
      read!(h_BatchFile,g)
      read!(h_BatchFile,b)

      Dataset[1:Pixels,sampleIdx] = convert(Array{Float64,2},r)./256
      Dataset[Pixels+1:2*Pixels,sampleIdx] = convert(Array{Float64,2},g)./256
      Dataset[2*Pixels+1:end,sampleIdx] = convert(Array{Float64,2},b)./256
    end

    # Close File
    close(h_BatchFile)

    # Get Label Names
    LabelNames = labelnames()

    # Conversion to Int type
    # Dataset = convert(Array{Int},Dataset)
    Labels = convert(Array{Int},Labels)

    # Check for grey
    if grey
      Dataset = toLuminance(Dataset)
    end

    # Apply Normalization
    if normalize_images
      Dataset = normalize(Dataset)
    end
  else
    Features = grey ? Pixels : 3*Pixels
    Dataset = Array(Float64,Features,0)
    Labels  = []
    for i= 1:5
      D, L, LabelNames = traindata(batch_number=i,normalize_images=normalize_images,grey=grey)
      Dataset = hcat(Dataset,D)
      Labels  = [Labels;L]
    end
  end

  return Dataset,Labels,LabelNames
end


function testdata(;normalize_images=false,grey=false)
  DataDir = joinpath(dirname(@__FILE__),"..","data","bin")
  BatchFile = "test_batch.bin"
  BatchFile = joinpath(DataDir,BatchFile)

  # Fixed dataset values
  Pixels = 1024
  Features = 3*Pixels
  Samples = 10000
  Labels = 10

  # Initialize Dataset Structures
  Dataset = Array(UInt8,Features,Samples)
  Labels = Array(UInt8,Samples,1)
  LabelNames = Array(AbstractString,10,1)

  # Open Batch File
  h_BatchFile = open(BatchFile)

  # Loop Over Samples
  r = Array(UInt8,Pixels,1)
  g = Array(UInt8,Pixels,1)
  b = Array(UInt8,Pixels,1)
  for sampleIdx = 1:Samples
    # Read Label Byte
    Labels[sampleIdx] = read(h_BatchFile,UInt8)+1
    # Read Color Byte Arrays
    read!(h_BatchFile,r)
    read!(h_BatchFile,g)
    read!(h_BatchFile,b)
    # Set Features in Dataset
    Dataset[1:Pixels,sampleIdx] = r
    Dataset[Pixels+1:2*Pixels,sampleIdx] = g
    Dataset[2*Pixels+1:end,sampleIdx] = b
  end

  # Close File
  close(h_BatchFile)

  # Get Label Names
  LabelNames = labelnames()

  # Conversion to Int type
  Dataset = convert(Array{Int},Dataset)
  Labels = convert(Array{Int},Labels)

  # Check for grey
  if grey
    Dataset = toLuminance(Dataset)
  end

  # Apply Normalization
  if normalize_images
    Dataset = normalize(Dataset)
  end

  return Dataset, Labels, LabelNames
end

end #module
