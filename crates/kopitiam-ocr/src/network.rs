//! Ported from Tesseract `src/lstm/network.{cpp,h}` (commit db0ec62, Apache-2.0,
//! © 2013 Google Inc., Author: Ray Smith; part of the Tesseract project),
//! translated to Rust for KOPITIAM (AGPL-3.0-only). Close adaptation: the
//! [`NetworkType`] tag enum, the `kTypeNames` string table, the common per-layer
//! header, and the `CreateFromFile` node-type dispatch follow Tesseract exactly;
//! re-expressed with a Rust trait object in place of the C++ class hierarchy. See
//! docs/ACKNOWLEDGEMENTS.md.
//!
//! # What this is
//!
//! This module is the graph loader for the LSTM recognizer network. A
//! `.traineddata` `Lstm` component is a serialized tree of layers; [`Network`]
//! reads that tree from a [`TFile`] ([`Network::deserialize`]) and runs a forward
//! pass over an activation buffer ([`Network::forward`]), producing the
//! per-timestep output over the recoded alphabet (softmax logits) that feeds the
//! beam decoder (phase 5).
//!
//! Every concrete layer implements [`NetworkNode`]; [`create_from_file`] is the
//! factory that reads the shared header (`CreateFromFile`, network.cpp:214),
//! decides the concrete type, and delegates to that type's `deserialize` for its
//! payload. Composite layers ([`Series`](crate::series::Series),
//! [`Parallel`](crate::parallel::Parallel),
//! [`Reversed`](crate::reversed::Reversed)) recurse back into
//! [`create_from_file`].
//!
//! # Node types: implemented vs deferred (forward)
//!
//! Every node type below is **deserialized** (so an arbitrary real recognizer
//! graph loads and its output dimension is known). Forward evaluation is
//! implemented for the 1-D text-line path — [`Input`](crate::input::Input),
//! [`Series`](crate::series::Series), [`Parallel`](crate::parallel::Parallel),
//! [`Reversed`](crate::reversed::Reversed) (x-reversal for BiLSTM),
//! [`FullyConnected`](crate::fullyconnected::FullyConnected) (all nonlinearities
//! incl. softmax), the 1-D [`Lstm`](crate::lstm::Lstm), and the general-stride
//! [`Reconfig`](crate::reconfig::Reconfig)/[`Maxpool`](crate::maxpool::Maxpool)/
//! [`Convolve`](crate::convolve::Convolve). The 2-D and softmax-feedback LSTM
//! *forward* variants are deferred (they surface a clear error); their
//! deserialization is complete. `NT_TENSORFLOW` is unsupported, as in Tesseract.

use std::fmt;

use crate::error::{Error, Result};
use crate::networkio::NetworkIO;
use crate::serialis::TFile;

/// Run-time type tag of a network layer. Ordinals and names must stay in sync
/// with [`TYPE_NAMES`]. Tesseract: `enum NetworkType` (network.h:40).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i8)]
pub enum NetworkType {
    /// The naked base class (invalid as a concrete layer).
    None = 0,
    /// Inputs from an image.
    Input,
    /// Duplicates inputs in a sliding-window neighborhood.
    Convolve,
    /// Chooses the max result from a rectangle.
    Maxpool,
    /// Runs networks in parallel.
    Parallel,
    /// Runs identical networks in parallel.
    Replicated,
    /// Runs LTR and RTL LSTMs in parallel.
    ParRlLstm,
    /// Runs Up and Down LSTMs in parallel.
    ParUdLstm,
    /// Runs 4 LSTMs in parallel.
    Par2dLstm,
    /// Executes a sequence of layers.
    Series,
    /// Scales the time/y size but makes the output deeper.
    Reconfig,
    /// Reverses the x direction of the inputs/outputs.
    XReversed,
    /// Reverses the y-direction of the inputs/outputs.
    YReversed,
    /// Transposes x and y.
    XyTranspose,
    /// Long-Short-Term-Memory block.
    Lstm,
    /// LSTM that only keeps its last output.
    LstmSummary,
    /// Fully connected logistic nonlinearity.
    Logistic,
    /// Fully connected rectified-linear version of logistic.
    Posclip,
    /// Fully connected rectified-linear version of tanh.
    Symclip,
    /// Fully connected with tanh nonlinearity.
    Tanh,
    /// Fully connected with rectifier nonlinearity.
    Relu,
    /// Fully connected with no nonlinearity.
    Linear,
    /// Softmax with exponential normalization, with CTC.
    Softmax,
    /// Softmax with exponential normalization, no CTC.
    SoftmaxNoCtc,
    /// 1-D LSTM with built-in fully connected softmax.
    LstmSoftmax,
    /// 1-D LSTM with built-in binary-encoded softmax.
    LstmSoftmaxEncoded,
    /// A TensorFlow graph (unsupported).
    TensorFlow,
}

/// The serialized string names of each [`NetworkType`], in ordinal order. Used
/// so that reordering the enum in a future Tesseract release cannot invalidate
/// existing model files. Tesseract: `kTypeNames` (network.cpp:60).
pub const TYPE_NAMES: [&str; 27] = [
    "Invalid",
    "Input",
    "Convolve",
    "Maxpool",
    "Parallel",
    "Replicated",
    "ParBidiLSTM",
    "DepParUDLSTM",
    "Par2dLSTM",
    "Series",
    "Reconfig",
    "RTLReversed",
    "TTBReversed",
    "XYTranspose",
    "LSTM",
    "SummLSTM",
    "Logistic",
    "LinLogistic",
    "LinTanh",
    "Tanh",
    "Relu",
    "Linear",
    "Softmax",
    "SoftmaxNoCTC",
    "LSTMSoftmax",
    "LSTMBinarySoftmax",
    "TensorFlow",
];

impl NetworkType {
    /// Maps an ordinal (`0..27`) to its [`NetworkType`].
    pub fn from_ordinal(value: i32) -> Option<NetworkType> {
        use NetworkType::*;
        Some(match value {
            0 => None,
            1 => Input,
            2 => Convolve,
            3 => Maxpool,
            4 => Parallel,
            5 => Replicated,
            6 => ParRlLstm,
            7 => ParUdLstm,
            8 => Par2dLstm,
            9 => Series,
            10 => Reconfig,
            11 => XReversed,
            12 => YReversed,
            13 => XyTranspose,
            14 => Lstm,
            15 => LstmSummary,
            16 => Logistic,
            17 => Posclip,
            18 => Symclip,
            19 => Tanh,
            20 => Relu,
            21 => Linear,
            22 => Softmax,
            23 => SoftmaxNoCtc,
            24 => LstmSoftmax,
            25 => LstmSoftmaxEncoded,
            26 => TensorFlow,
            _ => return Option::None,
        })
    }

    /// Looks up a [`NetworkType`] by its serialized string name.
    pub fn from_name(name: &str) -> Option<NetworkType> {
        TYPE_NAMES
            .iter()
            .position(|&n| n == name)
            .and_then(|i| NetworkType::from_ordinal(i as i32))
    }
}

/// The common per-layer header, read once by [`create_from_file`] before the
/// type-specific payload. Fields the recognizer never needs at inference (the
/// training flags) are kept as read for provenance but not acted on.
///
/// Tesseract: the locals at the top of `Network::CreateFromFile`
/// (network.cpp:214) plus `Network`'s common members.
#[derive(Clone, Debug)]
pub struct NetworkHeader {
    /// The concrete layer type.
    pub ntype: NetworkType,
    /// Whether the layer was serialized in the enabled-training state. A
    /// recognition model is not; this gates the weight-matrix training payload.
    pub training: bool,
    /// Whether the layer produces back-deltas (training only).
    pub needs_to_backprop: bool,
    /// Behavior control flags (`NetworkFlags`).
    pub network_flags: i32,
    /// Number of input values (`ni_`).
    pub ni: i32,
    /// Number of output values (`no_`). Some layers recompute this in their
    /// `deserialize` (Reconfig/Convolve/Maxpool/LSTM); the field is updated then.
    pub no: i32,
    /// Number of weights in this and sub-networks (`num_weights_`).
    pub num_weights: i32,
    /// The layer's unique name.
    pub name: String,
}

impl NetworkHeader {
    /// Reads the shared header from `fp`. Mirrors `getNetworkType`
    /// (network.cpp:191) followed by the field reads in `CreateFromFile`.
    fn read(fp: &mut TFile<'_>) -> Result<NetworkHeader> {
        // getNetworkType: first byte 0 => the type is written as a name string
        // (the modern format); non-zero => the ordinal directly (old format).
        let tag = fp.read_i8()?;
        let ntype = if tag == 0 {
            let type_name = fp.de_serialize_string()?;
            NetworkType::from_name(&type_name)
                .ok_or_else(|| Error::format(format!("invalid network layer type: {type_name}")))?
        } else {
            NetworkType::from_ordinal(tag as i32)
                .ok_or_else(|| Error::format(format!("invalid network layer ordinal: {tag}")))?
        };
        // TS_ENABLED == 1; anything else is treated as disabled.
        let training = fp.read_i8()? == 1;
        let needs_to_backprop = fp.read_i8()? != 0;
        let network_flags = fp.read_i32()?;
        let ni = fp.read_i32()?;
        let no = fp.read_i32()?;
        let num_weights = fp.read_i32()?;
        let name = fp.de_serialize_string()?;
        if ni < 0 || no < 0 || num_weights < 0 {
            return Err(Error::format(format!(
                "invalid network layer parameters: ni={ni} no={no} num_weights={num_weights}"
            )));
        }
        Ok(NetworkHeader {
            ntype,
            training,
            needs_to_backprop,
            network_flags,
            ni,
            no,
            num_weights,
            name,
        })
    }
}

/// A concrete network layer: knows its header (dimensions/type/name) and can run
/// a forward pass. The Rust stand-in for Tesseract's abstract `Network` class.
pub trait NetworkNode: fmt::Debug {
    /// The layer's header (type, dimensions, name).
    fn header(&self) -> &NetworkHeader;

    /// Runs forward propagation, producing the output activation buffer.
    /// Tesseract: `Network::Forward` (network.h:266).
    fn forward(&self, input: &NetworkIO) -> Result<NetworkIO>;

    /// The layer type.
    fn network_type(&self) -> NetworkType {
        self.header().ntype
    }

    /// Number of input features. Tesseract: `Network::NumInputs`.
    fn num_inputs(&self) -> i32 {
        self.header().ni
    }

    /// Number of output features. Tesseract: `Network::NumOutputs`.
    fn num_outputs(&self) -> i32 {
        self.header().no
    }

    /// The layer's name.
    fn name(&self) -> &str {
        &self.header().name
    }
}

/// Reads one network layer (and, recursively, its sub-layers) from `fp`,
/// dispatching on the deserialized type tag to the right concrete node.
///
/// Tesseract: `Network::CreateFromFile` (network.cpp:214). Returns an error
/// (rather than the C++ `nullptr`) on an unknown/unsupported type.
pub fn create_from_file(fp: &mut TFile<'_>) -> Result<Box<dyn NetworkNode>> {
    use NetworkType::*;
    let header = NetworkHeader::read(fp)?;
    let node: Box<dyn NetworkNode> = match header.ntype {
        Input => Box::new(crate::input::Input::deserialize(header, fp)?),
        Convolve => Box::new(crate::convolve::Convolve::deserialize(header, fp)?),
        Maxpool => Box::new(crate::maxpool::Maxpool::deserialize(header, fp)?),
        Reconfig => Box::new(crate::reconfig::Reconfig::deserialize(header, fp)?),
        Series => Box::new(crate::series::Series::deserialize(header, fp)?),
        Parallel | Replicated | ParRlLstm | ParUdLstm | Par2dLstm => {
            Box::new(crate::parallel::Parallel::deserialize(header, fp)?)
        }
        XReversed | YReversed | XyTranspose => {
            Box::new(crate::reversed::Reversed::deserialize(header, fp)?)
        }
        Lstm | LstmSoftmax | LstmSoftmaxEncoded | LstmSummary => {
            Box::new(crate::lstm::Lstm::deserialize(header, fp)?)
        }
        Softmax | SoftmaxNoCtc | Relu | Tanh | Linear | Logistic | Posclip | Symclip => Box::new(
            crate::fullyconnected::FullyConnected::deserialize(header, fp)?,
        ),
        TensorFlow => return Err(Error::format("unsupported TensorFlow model")),
        None => return Err(Error::format("invalid network layer type: None")),
    };
    Ok(node)
}

/// The deserialized LSTM recognizer network: a tree of [`NetworkNode`]s rooted
/// at a single top-level layer (typically a [`Series`](crate::series::Series)).
///
/// Construct with [`Network::deserialize`] over the `Lstm` component of a
/// `.traineddata` file (see [`crate::tessdata::TessdataManager`]), then
/// [`Network::forward`] an [`NetworkIO`] through it to get per-timestep logits.
#[derive(Debug)]
pub struct Network {
    root: Box<dyn NetworkNode>,
}

impl Network {
    /// Deserializes a whole recognizer network from `fp` (positioned at the
    /// start of the `Lstm` component). Tesseract: `Network::CreateFromFile`
    /// applied to the top-level layer (`lstmrecognizer.cpp` `DeSerialize`).
    pub fn deserialize(fp: &mut TFile<'_>) -> Result<Network> {
        Ok(Network {
            root: create_from_file(fp)?,
        })
    }

    /// Runs a forward pass, returning the output activation buffer — the
    /// per-timestep output over the recoded alphabet.
    pub fn forward(&self, input: &NetworkIO) -> Result<NetworkIO> {
        self.root.forward(input)
    }

    /// The network's output feature dimension (softmax code range).
    pub fn num_outputs(&self) -> i32 {
        self.root.num_outputs()
    }

    /// The network's input feature dimension.
    pub fn num_inputs(&self) -> i32 {
        self.root.num_inputs()
    }

    /// The top-level layer's type.
    pub fn network_type(&self) -> NetworkType {
        self.root.network_type()
    }

    /// The top-level layer's name.
    pub fn name(&self) -> &str {
        self.root.name()
    }

    /// The root node, for callers that need to inspect the graph.
    pub fn root(&self) -> &dyn NetworkNode {
        self.root.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn type_names_round_trip_through_ordinals() {
        for (i, &name) in TYPE_NAMES.iter().enumerate() {
            let by_ord = NetworkType::from_ordinal(i as i32).unwrap();
            let by_name = NetworkType::from_name(name).unwrap();
            assert_eq!(by_ord, by_name);
        }
    }

    #[test]
    fn unknown_type_name_and_ordinal_are_rejected() {
        assert!(NetworkType::from_name("NotAType").is_none());
        assert!(NetworkType::from_ordinal(999).is_none());
        assert!(NetworkType::from_ordinal(-5).is_none());
    }

    #[test]
    fn lstm_type_ordinal_is_14() {
        // Guards the enum ordering against the serialized-name table.
        assert_eq!(NetworkType::Lstm as i32, 14);
        assert_eq!(TYPE_NAMES[14], "LSTM");
        assert_eq!(NetworkType::Softmax as i32, 22);
        assert_eq!(TYPE_NAMES[22], "Softmax");
    }
}
