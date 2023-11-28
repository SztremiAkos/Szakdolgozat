import WrapperPage from "../pages/home/WrapperPage";
import { RouteType } from "./config";

const appRoutes: RouteType[] = [
  {
    element: <WrapperPage />,
    state: "predict",
    path: "predict",
  },
];
export default appRoutes;
