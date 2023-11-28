import { useState } from "react";
import { DefaultResponse, PredictRequest, predict } from "../../api/predict";
import PredictPage from "./PredictPage";
import ResultsPage from "./ResultsPage";

const WrapperPage = () => {
  const [selectedSystem, setSelectedSystem] = useState("");
  const [isPredictDone, setPredictIsDone] = useState(false);
  const [results, setResults] = useState(DefaultResponse);
  const [isLoading, setIsloading] = useState(false);

  const onPredict = async (modelParams: number[]) => {
    const req: PredictRequest = {
      name: selectedSystem,
      modelParams,
    };
    setIsloading(true);
    setResults(await predict(req));
    setIsloading(false);
    setPredictIsDone(true);
  };
  const onBack = () => {
    setPredictIsDone(false);
    setSelectedSystem("");
  }
  const updateSelectedSystem = (system: string) => setSelectedSystem(system);

  return isPredictDone ? (
    <ResultsPage
      result={results}
      selectedSystem={selectedSystem}
      back={onBack}
    />
  ) : (
    <PredictPage
      isLoading={isLoading}
      setSelectedSystem={updateSelectedSystem}
      onPredict={onPredict}
      selectedSystem={selectedSystem}
    />
  );
};

export default WrapperPage;
