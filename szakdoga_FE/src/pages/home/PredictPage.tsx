import { Autocomplete, Box, Button, TextField } from "@mui/material";
import systems from "../../configs/systemConfig";
import { RefObject, createRef, useEffect, useState } from "react";
import Loader from "../../components/Loader";
import Page from "../../components/Page";

type Props = {
  setSelectedSystem: (system: string) => void;
  onPredict: (params: number[]) => void;
  selectedSystem: string;
  isLoading: boolean;
};
const PredictPage = (props: Props) => {
  const { setSelectedSystem, onPredict, selectedSystem, isLoading } = props;

  const [inputFields, setInputFields] = useState<string[]>([]);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const [inputRefs, setInputRefs] = useState<RefObject<any>[]>([]);

  useEffect(() => {
    const system = systems.filter((a) => a.name === selectedSystem)[0];
    if (system) {
      setInputFields(system.paramLabels);
    } else {
      setInputFields([]);
    }

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    setInputRefs((elRefs: any) =>
      Array(inputFields.length)
        .fill(0)
        .map((_, i) => elRefs[i] || createRef())
    );
  }, [inputFields.length, selectedSystem]);
  return (
    <>
      {isLoading && <Loader />}
      <Page title="Predict">
        <Box>
          <Box
            sx={{
              justifyContent: "center",
              gap: 2,
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
            }}
          >
            <Autocomplete
              disablePortal
              id="combo-box-demo"
              onChange={(_, systemName) => {
                setSelectedSystem(systemName ?? "");
              }}
              sx={{ width: 300 }}
              options={systems.map((system) => system.name)}
              renderInput={(params) => <TextField {...params} label="System" />}
            />
            {inputFields.map((label, index) => {
              return (
                <TextField
                  key={index}
                  type="number"
                  label={label}
                  inputRef={inputRefs[index]}
                  sx={{ width: 300 }}
                ></TextField>
              );
            })}
            {selectedSystem && (
              <>
                <Button
                  variant="contained"
                  onClick={() =>
                    onPredict(inputRefs.map((ref) => ref.current.value))
                  }
                >
                  Predict
                </Button>
              </>
            )}
          </Box>
        </Box>
      </Page>
    </>
  );
};

export default PredictPage;